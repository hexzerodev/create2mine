use fs2::FileExt;
use hex::FromHex;
use itertools::Itertools;
use rand::{distributions::Standard, thread_rng, Rng};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use tiny_keccak::{Hasher, Keccak};

/* -------------------------------------------------------------------------- */
/*                                   config                                   */
/* -------------------------------------------------------------------------- */
pub struct Config {
    pub factory_address: [u8; 20],
    pub calling_address: [u8; 20],
    pub init_code_hash: [u8; 32],
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

        //
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

        Ok(Config {
            factory_address,
            calling_address,
            init_code_hash,
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
    const ZERO_CHARACTER: u8 = 0x00;

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
                .filter(|&n| *n == ZERO_CHARACTER)
                .count();

            // <= 2 zero bytes, return
            if zero_bytes_count <= 2 {
                return;
            }

            // calculate leading zeros
            let leading_zero_count = {
                let mut c = 0;
                for (i, b) in hash_res.iter().dropping(12).enumerate() {
                    if *b != ZERO_CHARACTER {
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
