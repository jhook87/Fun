import { readFile } from 'fs/promises';
import { uploadToIPFS } from '../utils/ipfs';
import * as path from 'path';

async function main() {
  const args = process.argv.slice(2);
  if (!args[0]) {
    console.error('Usage: ts-node scripts/pin.ts <file>');
    process.exit(1);
  }
  const filePath = args[0];
  const data = await readFile(filePath);
  const name = path.basename(filePath);
  try {
    const uri = await uploadToIPFS(name, data);
    console.log('Pinned to IPFS:', uri);
  } catch (err) {
    console.error('Failed to pin file:', err);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
