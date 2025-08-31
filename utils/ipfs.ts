import { Web3Storage, File as W3File } from 'web3.storage';
import * as dotenv from 'dotenv';

// Load environment variables from .env
dotenv.config();

// Get your Web3.Storage API token from the environment
const token = process.env.WEB3_STORAGE_TOKEN;
if (!token) {
  throw new Error('WEB3_STORAGE_TOKEN is not provided');
}

// Initialize Web3.Storage client
const client = new Web3Storage({ token });

/**
 * Upload a file or data buffer to Web3.Storage and return an IPFS URI.
 *
 * @param name - The filename to use for the uploaded file
 * @param content - The content to upload; can be a string or Uint8Array
 * @returns The IPFS URI of the uploaded content
 */
export async function uploadToIPFS(name: string, content: Uint8Array | string): Promise<string> {
  const data = typeof content === 'string' ? new TextEncoder().encode(content) : content;
  const file = new W3File([data], name);
  const cid = await client.put([file]);
  return `ipfs://${cid}/${encodeURIComponent(name)}`;
}
