use byteorder::ByteOrder;
use console::Term;
use fs2::FileExt;
use hex::FromHex;
use itertools::Itertools;
use ocl::DeviceType;
use rand::{distributions::Standard, thread_rng, Rng};
use rayon::prelude::*;
use separator::Separatable;
use std::fs::{self, OpenOptions};
use std::{io::Write, time};
use tiny_keccak::{Hasher, Keccak};

/* -------------------------------------------------------------------------- */
/*                                  constants                                 */
/* -------------------------------------------------------------------------- */
static EFFICIENT_ADDRESS_CL_SRC: &'static str = include_str!("./kernels/efficientAddress.cl");

const DEFAULT_MIN_LEADING_ZEROES: u8 = 3;
const DEFAULT_MIN_ZEROES_COUNT: u8 = 5;
const DEFAULT_FALLBACK_TO_CPU: bool = true;

const CONTROL_CHARACTERS: u8 = 0xff;
const EIGHT_ZERO_BYTES: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
const ZERO_BYTE: u8 = 0x00;

const WORK_SIZE: usize = 0x4000000; // max. 0x15400000 to abs. max 0xffffffff

const INTERVAL: u64 = 0x100000000;

/* -------------------------------------------------------------------------- */
/*                                   config                                   */
/* -------------------------------------------------------------------------- */
pub struct Config {
    pub factory_address: [u8; 20],
    pub calling_address: [u8; 20],
    pub init_code_hash: [u8; 32],
    pub min_leading_zeroes: u8,
    pub min_zeroes_count: u8,
    pub fallback_to_cpu: bool,
}

impl Config {
    pub fn new(mut args: std::env::Args) -> Result<Self, String> {
        // get args, skip first arg (program name)
        args.next();

        // read args
        let factory_address_string = match args.next() {
            Some(arg) => arg,
            None => return Err("Missing factory address".to_string()),
        };

        let calling_address_string = match args.next() {
            Some(arg) => arg,
            None => return Err("Missing calling address".to_string()),
        };

        let init_code_hash_string = match args.next() {
            Some(arg) => arg,
            None => return Err("Missing init code hash".to_string()),
        };

        let min_leading_zeroes_string = match args.next() {
            Some(arg) => arg,
            None => format!("{}", DEFAULT_MIN_LEADING_ZEROES),
        };

        let min_zeroes_count_string = match args.next() {
            Some(arg) => arg,
            None => format!("{}", DEFAULT_MIN_ZEROES_COUNT),
        };

        let fallback_to_cpu_string = match args.next() {
            Some(arg) => arg,
            None => format!("{}", DEFAULT_FALLBACK_TO_CPU),
        };

        // parse hex strings
        let factory_address = match Self::parse_address(&factory_address_string) {
            Ok(a) => a,
            Err(e) => return Err(format!("invalid factory address: {}", e)),
        };
        let calling_address = match Self::parse_address(&calling_address_string) {
            Ok(a) => a,
            Err(e) => return Err(format!("invalid calling address: {}", e)),
        };
        let init_code_hash = match Self::parse_hash(&init_code_hash_string) {
            Ok(a) => a,
            Err(e) => return Err(format!("invalid init code hash: {}", e)),
        };

        // parse numbers
        let min_leading_zeroes = match min_leading_zeroes_string.parse() {
            Ok(t) => t,
            Err(e) => return Err(format!("invalid min_leading_zeroes arg: {}", e)),
        };

        let min_zeroes_count = match min_zeroes_count_string.parse() {
            Ok(t) => t,
            Err(e) => return Err(format!("invalid min_zeroes_count arg: {}", e)),
        };

        // parse bool
        let fallback_to_cpu = match fallback_to_cpu_string.parse() {
            Ok(t) => t,
            Err(e) => return Err(format!("invalid fallback_to_cpu arg: {}", e)),
        };

        Ok(Config {
            factory_address,
            calling_address,
            init_code_hash,
            min_leading_zeroes,
            min_zeroes_count,
            fallback_to_cpu,
        })
    }

    fn parse_address(addr_str: &str) -> Result<[u8; 20], String> {
        // convert to string
        let mut s = addr_str.to_string();

        // strip leading 0x
        if (addr_str).starts_with("0x") {
            s = addr_str
                .char_indices()
                .nth(2)
                .and_then(|(i, _)| addr_str.get(i..))
                .unwrap()
                .to_string();
        }

        // parse hex
        let v = match Vec::from_hex(&s) {
            Ok(a) => a,
            // Err(e) => return Err(Box::new(e)),
            Err(e) => return Err(format!("invalid hex number: {}", e)),
        };

        // check length
        if v.len() != 20 {
            return Err("invalid length".to_string());
        }

        // to fixed array size
        let mut array = [0; 20];
        let bytes = &v[..array.len()];
        array.copy_from_slice(bytes);

        Ok(array)
    }

    fn parse_hash(addr_str: &str) -> Result<[u8; 32], String> {
        // convert to string
        let mut s = addr_str.to_string();

        // strip leading 0x
        if (addr_str).starts_with("0x") {
            s = addr_str
                .char_indices()
                .nth(2)
                .and_then(|(i, _)| addr_str.get(i..))
                .unwrap()
                .to_string();
        }

        // parse hex
        let v = match Vec::from_hex(&s) {
            Ok(a) => a,
            Err(e) => return Err(format!("invalid hex number: {}", e)),
        };

        // check length
        if v.len() != 32 {
            return Err("invalid length".to_string());
        }

        // to fixed array size
        let mut array = [0; 32];
        let bytes = &v[..array.len()];
        array.copy_from_slice(bytes);

        Ok(array)
    }
}

/* -------------------------------------------------------------------------- */
/*                                     run                                    */
/* -------------------------------------------------------------------------- */
/// Given a Config object with a factory address, a caller address, and a
/// keccak-256 hash of the contract initialization code, search for salts that
/// will enable the factory contract to deploy a contract to a gas-efficient
/// address via CREATE2.
///
/// A CREATE2 destination address is calculated as follows:
/// ```
/// initialisation_code = memory[offset:offset+size]
/// address = keccak256(0xff + sender_address + salt + keccak256(initialisation_code))[12:]
/// ```
///
/// The 32-byte salt is constructed as follows:
///   - the 20-byte calling address (to prevent frontrunning)
///   - a random 6-byte segment (to prevent collisions with other runs)
///   - a 6-byte nonce segment (incrementally stepped through during the run)
///
/// When a salt that will result in the creation of a gas-efficient contract
/// address is found, it will be appended to `efficient_addresses.txt` along
/// with the resultant address and the "value" (i.e. approximate rarity) of the
/// resultant address.
pub fn run(config: Config) {
    // (create if necessary) and open a file where found salts will be written
    let file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("efficient_addresses.txt")
        .expect("Could not create or open `efficient_addresses.txt` file.");

    let initial_control_character: Vec<u8> = vec![0xff];
    let footer: [u8; 32] = config.init_code_hash;
    const MAX_INCREMENTER: u64 = 281474976710655; // 2 ** 48 - 1 (6 bytes max)

    // loop {

    // get 6-byte nonce
    let rng = thread_rng();
    let salt_random_segment = rng.sample_iter(Standard).take(6).collect::<Vec<u8>>();

    // header: 0xff ++ factory ++ caller ++ salt_random_segment (47 bytes)
    let header = {
        let mut header_vec: Vec<u8> = vec![];
        header_vec.extend(&initial_control_character);
        header_vec.extend(config.factory_address.iter());
        header_vec.extend(config.calling_address.iter());
        header_vec.extend(salt_random_segment);

        // to fixed array size
        let mut array = [0; 47];
        let bytes = &header_vec[..array.len()];
        array.copy_from_slice(bytes);
        array
    };

    // create new hash object
    let mut hash_header = Keccak::v256();

    // hash to derive address
    hash_header.update(&header);

    // iterate over salt incrementally
    (0..MAX_INCREMENTER)
        .into_par_iter()
        .map(|x| u64_to_fixed_6(&x))
        .for_each(|salt_incremented_segment| {
            // calculate address
            let hash_res: [u8; 32] = {
                // clone the partially-hashed object
                let mut hash = hash_header.clone();

                hash.update(&salt_incremented_segment);
                hash.update(&footer);

                let mut res: [u8; 32] = [0; 32];
                hash.finalize(&mut res);

                res
            };

            // get zero bytes count
            let zero_bytes_count = hash_res
                .iter()
                .dropping(12)
                .filter(|&n| *n == ZERO_BYTE)
                .count();

            // <= 2 zero bytes, return
            if zero_bytes_count <= 2 {
                return;
            }

            // calculate leading zeros
            let leading_zero_count = {
                let mut c = 0;
                for (i, b) in hash_res.iter().dropping(12).enumerate() {
                    if *b != ZERO_BYTE {
                        c = i;
                        break;
                    }
                }
                c
            };

            // reward amount
            if leading_zero_count < 4 {
                return;
            }

            // truncate the first 12 bytes from the hash to derive the address
            let address_bytes = {
                let mut address_bytes: [u8; 20] = Default::default();
                address_bytes.copy_from_slice(&hash_res[12..]);

                address_bytes
            };

            // get address
            let address = {
                let address_hex_string = hex::encode(&address_bytes);
                format!("{}", &address_hex_string)
            };

            // get full salt
            let full_salt = {
                let header_hex_string = hex::encode(&header);
                let body_hex_string = hex::encode(salt_incremented_segment.to_vec());
                format!("0x{}{}", &header_hex_string[42..], &body_hex_string)
            };

            // checksummed address
            let checksum_address = {
                let address_encoded = address.as_bytes();

                // create new hash object for computing the checksum
                let mut checksum_hash = Keccak::v256();

                // update with utf8-encoded address (total: 20 bytes)
                checksum_hash.update(&address_encoded);

                // hash the payload and get the result
                let mut checksum_res: [u8; 32] = [0; 32];
                checksum_hash.finalize(&mut checksum_res);
                let address_hash = hex::encode(checksum_res);

                // compute the checksum using the above hash
                let mut checksum_address = "0x".to_string();
                for nibble in 0..address.len() {
                    let hash_character = i64::from_str_radix(
                        &address_hash.chars().nth(nibble).unwrap().to_string(),
                        16,
                    )
                    .unwrap();
                    let character = address.chars().nth(nibble).unwrap();
                    if hash_character > 7 {
                        checksum_address = format!(
                            "{}{}",
                            checksum_address,
                            character.to_uppercase().to_string()
                        );
                    } else {
                        checksum_address = format!("{}{}", checksum_address, character.to_string());
                    }
                }

                checksum_address
            };

            // display salt and address
            let output = format!(
                "{} => {} => {}",
                full_salt,
                checksum_address,
                leading_zero_count * 20 + zero_bytes_count
            );
            println!("{}", &output);

            // write to file
            {
                file.lock_exclusive().expect("Couldn't lock file.");

                writeln!(&file, "{}", &output)
                    .expect("Couldn't write to `efficient_addresses.txt` file.");

                file.unlock().expect("Couldn't unlock file.")
            }
        })
}

/// Given a Config object with a factory address, a caller address, a keccak-256
/// hash of the contract initialization code, search for salts using OpenCL that
/// will enable the factory contract to deploy a contract to a gas-efficient
/// address via CREATE2. This method also takes threshold values for both leading
/// zero bytes and total zero bytes - any address that does not meet or exceed
/// the threshold will not be returned. Default threshold values are three leading
/// zeroes or five total zeroes.
///
/// The 32-byte salt is constructed as follows:
///   - the 20-byte calling address (to prevent frontrunning)
///   - a random 4-byte segment (to prevent collisions with other runs)
///   - a 4-byte segment unique to each work group running in parallel
///   - a 4-byte nonce segment (incrementally stepped through during the run)
///
/// When a salt that will result in the creation of a gas-efficient contract
/// address is found, it will be appended to `efficient_addresses.txt` along
/// with the resultant address and the "value" (i.e. approximate rarity) of the
/// resultant address.
pub fn gpu(config: Config) -> ocl::Result<()> {
    let platforms = ocl::Platform::list();

    // check platforms
    if platforms.len() <= 0 {
        panic!("No platforms found");
    }

    // print platform & device info
    println!("\nPlatforms\n--------------------------------------------------------------------------------");
    platforms.iter().for_each(|p| {
        let name = p.name().unwrap_or("".to_string());
        let profile = p.profile().unwrap_or("".to_string());
        let vendor = p.vendor().unwrap_or("".to_string());

        println!("NAME: {}\nPROFILE: {}\nVENDOR: {}", name, profile, vendor);

        let devices = match ocl::Device::list(p, Some(DeviceType::GPU)) {
            Ok(r) => r,
            Err(e) => panic!("failed to list devices: {}", e),
        };

        println!("\n\tGPU Devices\n\t--------------------------------------------------------------------------------");
        devices.iter().for_each(|d| {
            let name = d.name().unwrap_or("".to_string());
            let address_bit = match d.info(ocl::enums::DeviceInfo::AddressBits) {
                Ok(r) => r,
                Err(_e) => ocl::enums::DeviceInfoResult::AddressBits(0),
            };
            let extensions = match d.info(ocl::enums::DeviceInfo::Extensions) {
                Ok(r) => r,
                Err(_e) => ocl::enums::DeviceInfoResult::Extensions("".to_string()),
            };
            println!(
                "\tNAME: {}\n\tADDRESS_BITS: {}\n\tEXTENSIONS: {}",
                name, address_bit, extensions
            );
        });
    });
    println!(
        "\n--------------------------------------------------------------------------------\n"
    );

    //
    // _add()?;
    // _reduction()?;
    // _reduction_vector()?;
    _mint(config)?;
    // _t()?;

    return Ok(());
}

fn _mint(config: Config) -> ocl::Result<()> {
    // (create if necessary) and open a file where found salts will be written
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("efficient_addresses.txt")
        .expect("Could not create or open `efficient_addresses.txt` file.");

    let factory: [u8; 20] = config.factory_address;
    let caller: [u8; 20] = config.calling_address;
    let init_hash: [u8; 32] = config.init_code_hash;

    let term = Term::stdout();
    let rng = thread_rng();

    let start_time: f64 = time::SystemTime::now()
        .duration_since(time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as f64;

    let mut rate: f64 = 0.0;
    let mut cumulative_nonce: u64 = 0;
    let mut view_buf = [0; 8];

    let mut found: u64 = 0;
    let mut found_list: Vec<String> = vec![];

    let min_leading_zero_count = config.min_leading_zeroes;
    let min_zero_bytes_count = config.min_zeroes_count;

    // platform, devices
    let platform = ocl::Platform::default();
    let devices = match ocl::Device::list(platform, Some(DeviceType::GPU)) {
        Ok(r) => r,
        Err(e) => panic!("failed to list devices: {}", e),
    };
    let device = if devices.len() > 0 {
        devices[0]
    } else {
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!("No GPU found, falling back on CPU");
        println!(
            "--------------------------------------------------------------------------------"
        );
        ocl::Device::first(platform).unwrap()
    };

    // context, queue
    let context = ocl::Context::builder().devices(device.clone()).build()?;
    let queue = ocl::Queue::new(&context, device, None)?;

    // program
    let program = ocl::Program::builder()
        .devices(device)
        .src(EFFICIENT_ADDRESS_CL_SRC)
        .build(&context)?;

    //
    loop {
        // data
        // create a random 4-byte salt using the random number generator
        let salt = rng
            .clone()
            .sample_iter(Standard)
            .take(4)
            .collect::<Vec<u8>>();
        let message_buffer = {
            let mut message_vec = vec![CONTROL_CHARACTERS];
            message_vec.extend(factory.iter());
            message_vec.extend(caller.iter());
            message_vec.extend(&salt);
            message_vec.extend(EIGHT_ZERO_BYTES.iter());
            message_vec.extend(init_hash.iter());
            let message: [u8; 85] = to_fixed_85(&message_vec);

            // build a corresponding buffer for passing the message to the kernel
            let buffer = ocl::Buffer::builder()
                .queue(queue.clone())
                .flags(ocl::flags::MEM_READ_ONLY)
                .len(message.len())
                .copy_host_slice(&message)
                .build()?;

            buffer
        };

        let target: [u8; 2] = [min_leading_zero_count, min_zero_bytes_count];
        let target_buffer = {
            let buffer = ocl::Buffer::builder()
                .queue(queue.clone())
                .flags(ocl::flags::MEM_READ_ONLY)
                .len(target.len())
                .copy_host_slice(&target)
                .build()?;

            buffer
        };

        let mut nonce: [u64; 1] = [0];
        let mut nonce_buffer = {
            let buffer = ocl::Buffer::builder()
                .queue(queue.clone())
                .flags(ocl::flags::MEM_READ_ONLY)
                .len(nonce.len())
                .copy_host_slice(&nonce)
                .build()?;

            buffer
        };

        // output
        let mut solutions: Vec<u64> = vec![0; 256];
        let solutions_buffer = {
            let buffer = ocl::Buffer::builder()
                .queue(queue.clone())
                .flags(ocl::flags::MEM_WRITE_ONLY)
                .len(solutions.len())
                .copy_host_slice(&solutions)
                .build()?;

            buffer
        };

        let mut solution_count: Vec<u32> = vec![0];
        let solution_count_buffer = {
            let buffer = ocl::Buffer::builder()
                .queue(queue.clone())
                .flags(ocl::flags::MEM_WRITE_ONLY)
                .len(1)
                .fill_val(0u32)
                .build()?;
            buffer
        };

        // run kernel, incrementing nonce
        loop {
            // term display
            {
                // clear the terminal screen
                term.clear_screen()?;

                // calculate the current time
                let current_time: f64 = time::SystemTime::now()
                    .duration_since(time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as f64;

                // get the total runtime and parse into hours : minutes : seconds
                let total_runtime = current_time - start_time;
                let total_runtime_hrs = *&total_runtime as u64 / (3600);
                let total_runtime_mins = (*&total_runtime as u64 - &total_runtime_hrs * 3600) / 60;
                let total_runtime_secs = &total_runtime
                    - (&total_runtime_hrs * 3600) as f64
                    - (&total_runtime_mins * 60) as f64;

                // determine the number of attempts being made per second
                if total_runtime > 0.0 {
                    rate = 1.0 / total_runtime;
                }
                let work_rate: f64 =
                    (((WORK_SIZE as u128 / 1_000_000) * (cumulative_nonce as u128)) as f64) * rate;

                // fill the buffer for viewing the properly-formatted nonce
                byteorder::LittleEndian::write_u64(&mut view_buf, nonce[0]);

                // calculate the terminal height, defaulting to a height of ten rows
                let size = terminal_size::terminal_size();
                let height: u16;
                if let Some((terminal_size::Width(_w), terminal_size::Height(h))) = size {
                    height = h;
                } else {
                    height = 10;
                }

                // display information about the total runtime and work size
                term.write_line(&format!(
                    "total runtime: {}:{:02}:{:02} ({} cycles)\t\t\t\
              work size per cycle: {}",
                    total_runtime_hrs,
                    total_runtime_mins,
                    total_runtime_secs,
                    cumulative_nonce,
                    WORK_SIZE.separated_string()
                ))?;

                // display information about the attempt rate and found solutions
                term.write_line(&format!(
                    "rate: {:.2} million attempts per second\t\t\t\
              total found this run: {}",
                    work_rate, &found
                ))?;

                // display information about the current search criteria
                term.write_line(&format!(
                    "current search space: {}xxxxxxxx{:08x}\t\t\
              threshold: {} leading or {} total zeroes",
                    hex::encode(&salt),
                    byteorder::BigEndian::read_u64(&view_buf),
                    target[0],
                    target[1]
                ))?;

                // display recently found solutions based on terminal height
                let rows: usize = if height < 5 { 1 } else { (height - 4) as usize };
                let last_rows: Vec<String> = found_list.iter().cloned().rev().take(rows).collect();
                let ordered: Vec<String> = last_rows.iter().cloned().rev().collect();
                let recently_found = &ordered.join("\n");
                term.write_line(&recently_found)?;
            }

            // run kernel
            {
                // build kernel
                let kernel = ocl::Kernel::builder()
                    .program(&program)
                    .name("hashMessage")
                    .arg(&message_buffer)
                    .arg(&target_buffer)
                    .arg(&nonce_buffer)
                    .arg(&solutions_buffer)
                    .arg(&solution_count_buffer)
                    .build()?;

                // enqueue kernel
                unsafe {
                    kernel
                        .cmd()
                        .queue(&queue)
                        .global_work_offset(kernel.default_global_work_offset())
                        .global_work_size(WORK_SIZE)
                        .enq()?;
                }
            }

            cumulative_nonce += 1;

            // read output
            {
                // read the number of solutions from the device
                solution_count_buffer.read(&mut solution_count).enq()?;

                // if at least one solution is found, end the loop
                if solution_count[0] != 0 {
                    break;
                }
            }

            // next loop
            {
                // if no solution has yet been found, increment the nonce
                nonce[0] += INTERVAL as u64;

                // update the nonce buffer with the incremented nonce value
                nonce_buffer = {
                    let buffer = ocl::Buffer::builder()
                        .queue(queue.clone())
                        .flags(ocl::flags::MEM_READ_ONLY)
                        .len(nonce.len())
                        .copy_host_slice(&nonce)
                        .build()?;

                    buffer
                };
            }
        }

        // output addresses
        {
            // read the located solutions from the device
            solutions_buffer.read(&mut solutions).enq()?;

            //
            solutions
                .iter()
                .filter(|&i| *i != 0)
                .map(|i| u64_to_le_fixed_8(i))
                .for_each(|solution| {
                    // check
                    if solution == EIGHT_ZERO_BYTES {
                        return;
                    }

                    // calculate address
                    let hash_res: [u8; 32] = {
                        let mut solution_message: Vec<u8> = vec![CONTROL_CHARACTERS];

                        solution_message.extend(factory.iter());
                        solution_message.extend(caller.iter());
                        solution_message.extend(salt.iter());
                        solution_message.extend(solution.iter());
                        solution_message.extend(init_hash.iter());

                        // create new hash object
                        let mut hash = Keccak::v256();

                        // update with header
                        hash.update(&solution_message);

                        // hash the payload and get the result
                        let mut res: [u8; 32] = [0; 32];
                        hash.finalize(&mut res);
                        res
                    };

                    // full salt
                    let full_salt = format!(
                        "0x{}{}{}",
                        hex::encode(&caller),
                        hex::encode(&salt),
                        hex::encode(&solution)
                    );

                    // process
                    let output = _process_hash(
                        &hash_res,
                        &full_salt,
                        &mut file,
                        min_zero_bytes_count as usize,
                        min_leading_zero_count as usize,
                    );

                    // update count
                    if let Some(_output) = output {
                        let show = format!("{} ({} / {})", &_output.0, &_output.1, &_output.2);
                        let next_found = vec![show.to_string()];
                        found_list.extend(next_found);
                        found += 1;
                    }
                })
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                                   helpers                                  */
/* -------------------------------------------------------------------------- */
/**
 * Convert a 64-bit unsigned integer to a fixed array of six bytes.
 */
fn u64_to_fixed_6(x: &u64) -> [u8; 6] {
    let mask: u64 = 0xff;
    let b1: u8 = ((x >> 40) & mask) as u8;
    let b2: u8 = ((x >> 32) & mask) as u8;
    let b3: u8 = ((x >> 24) & mask) as u8;
    let b4: u8 = ((x >> 16) & mask) as u8;
    let b5: u8 = ((x >> 8) & mask) as u8;
    let b6: u8 = (x & mask) as u8;
    [b1, b2, b3, b4, b5, b6]
}

/**
 *  Convert 64-bit unsigned integer to little-endian fixed array of eight bytes.
 */
fn u64_to_le_fixed_8(x: &u64) -> [u8; 8] {
    let mask: u64 = 0xff;
    let b1: u8 = ((x >> 56) & mask) as u8;
    let b2: u8 = ((x >> 48) & mask) as u8;
    let b3: u8 = ((x >> 40) & mask) as u8;
    let b4: u8 = ((x >> 32) & mask) as u8;
    let b5: u8 = ((x >> 24) & mask) as u8;
    let b6: u8 = ((x >> 16) & mask) as u8;
    let b7: u8 = ((x >> 8) & mask) as u8;
    let b8: u8 = (x & mask) as u8;
    [b8, b7, b6, b5, b4, b3, b2, b1]
}

/**
 * Convert a properly-sized vector to a fixed array of 85 bytes.
 */
fn to_fixed_85(bytes: &std::vec::Vec<u8>) -> [u8; 85] {
    let mut array = [0; 85];
    let bytes = &bytes[..array.len()];
    array.copy_from_slice(bytes);
    array
}

fn _process_hash(
    hash: &[u8; 32],
    full_salt: &str,
    file: &mut fs::File,
    min_zero_bytes_count: usize,
    min_leading_zero_count: usize,
) -> Option<(String, usize, usize)> {
    // get zero bytes count
    let zero_bytes_count = hash
        .iter()
        .dropping(12)
        .filter(|&n| *n == ZERO_BYTE)
        .count();

    // calculate leading zeros
    let leading_zero_count = {
        let mut c = 0;
        for (i, b) in hash.iter().dropping(12).enumerate() {
            if *b != ZERO_BYTE {
                c = i;
                break;
            }
        }
        c
    };

    if zero_bytes_count < min_zero_bytes_count && leading_zero_count < min_leading_zero_count {
        return None;
    }

    // truncate the first 12 bytes from the hash to derive the address
    let address_bytes = {
        let mut address_bytes: [u8; 20] = Default::default();
        address_bytes.copy_from_slice(&hash[12..]);

        address_bytes
    };

    // get address
    let address = {
        let address_hex_string = hex::encode(&address_bytes);
        format!("{}", &address_hex_string)
    };

    // checksummed address
    let checksum_address = {
        let address_encoded = address.as_bytes();

        // create new hash object for computing the checksum
        let mut checksum_hash = Keccak::v256();

        // update with utf8-encoded address (total: 20 bytes)
        checksum_hash.update(&address_encoded);

        // hash the payload and get the result
        let mut checksum_res: [u8; 32] = [0; 32];
        checksum_hash.finalize(&mut checksum_res);
        let address_hash = hex::encode(checksum_res);

        // compute the checksum using the above hash
        let mut checksum_address = "0x".to_string();
        for nibble in 0..address.len() {
            let hash_character =
                i64::from_str_radix(&address_hash.chars().nth(nibble).unwrap().to_string(), 16)
                    .unwrap();
            let character = address.chars().nth(nibble).unwrap();
            if hash_character > 7 {
                checksum_address = format!(
                    "{}{}",
                    checksum_address,
                    character.to_uppercase().to_string()
                );
            } else {
                checksum_address = format!("{}{}", checksum_address, character.to_string());
            }
        }

        checksum_address
    };

    // display salt and address
    let output = format!(
        "{} => {} => {}",
        full_salt,
        checksum_address,
        leading_zero_count * 20 + zero_bytes_count
    );
    println!("{}", &output);

    // write to file
    {
        file.lock_exclusive().expect("Couldn't lock file.");

        writeln!(file, "{}", &output).expect("Couldn't write to `efficient_addresses.txt` file.");

        file.unlock().expect("Couldn't unlock file.");
    }

    Some((output, leading_zero_count, zero_bytes_count))
}
