# create2mine

> A Rust program for finding salts that create gas-efficient Ethereum addresses via CREATE2.

Provide three arguments: a factory address (or contract that will call CREATE2), a caller address (for factory addresses that require it as a protection against frontrunning), and the keccak-256 hash of the initialization code of the contract that the factory will deploy.

For each efficient address found, the salt, resultant addresses, and value _(i.e. approximate rarity)_ will be written to `efficient_addresses.txt`. Be sure not to change the factory address or the init code without first removing any existing data to prevent the two salt types from becoming commingled.

# References

- https://github.com/0age/create2crunch
- https://medium.com/coinmonks/on-efficient-ethereum-addresses-3fef0596e263
