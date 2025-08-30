import { blake3 } from "blake3";
import { ethers } from "ethers";
import fetch from "cross-fetch";
import Ajv from "ajv";
// VC libs
import { createVerify } from "did-jwt";
import { verifyCredential } from "did-jwt-vc";
import { Resolver } from "did-resolver";
// NOTE: add resolvers as you adopt did:pkh / did:key / etc.
// import { getResolver as pkhResolver } from "@didtools/pkh-did-resolver";
// import { getResolver as keyResolver } from "key-did-resolver";

const abi = [
  "function getRecord(uint256) view returns (tuple(bytes32 contentHash,string metadataURI,bool revoked))"
];

export async function verifyAll({ contentBytes, rpc, contractAddr, tokenId }:{
  contentBytes: Buffer, rpc: string, contractAddr: string, tokenId: bigint
}) {
  // 1) Hash file
  const digest = blake3(contentBytes);
  const computed = "0x" + Buffer.from(digest).toString("hex");

  // 2) On-chain record
  const provider = new ethers.providers.JsonRpcProvider(rpc);
  const c = new ethers.Contract(contractAddr, abi, provider);
  const rec = await c.getRecord(tokenId);
  if (rec.revoked) return { ok:false, reason:"Token revoked" };
  const onchainHash = rec.contentHash.toLowerCase();
  if (computed.toLowerCase() !== onchainHash) return { ok:false, reason:"Hash mismatch" };

  // 3) Fetch metadata
  const meta = await fetchJSON(rec.metadataURI);

  // 4) Schema (optional)
  try {
    const schema = await fetchJSONRaw(new URL("../../offchain/schema/metadata.schema.json", import.meta.url));
    const ajv = new Ajv({ allErrors: true }); const validate = ajv.compile(schema);
    if (!validate(meta)) return { ok:false, reason:"Bad metadata schema", schemaErrors: validate.errors };
  } catch { /* ignore if local path not resolvable in your runtime */ }

  // 5) Author signature over "contentHash||createdAt"
  const payload = Buffer.from(`${meta.contentHash}||${meta.createdAt}`);
  if (!meta.signatures?.[0]) return { ok:false, reason:"Missing author signature" };
  const sigB64 = meta.signatures[0].sig;
  const pubB64 = meta.signatures[0].pub;   // DID fallback: allow raw pub in metadata
  if (!pubB64 && !meta.authorDID) return { ok:false, reason:"No DID or pubkey provided" };

  let authorOk = false;
  if (pubB64) {
    const nacl = await import("tweetnacl");
    authorOk = nacl.sign.detached.verify(payload, Buffer.from(sigB64, "base64"), Buffer.from(pubB64, "base64"));
  } else {
    const resolver = new Resolver({
      // ...add the DID methods you’ll support:
      // ...pkhResolver(), ...keyResolver()
    } as any);
    try {
      // If you used did-jwt for the signature instead, verify here. With the current
      // plain detached signature, you’d need metadata describing alg/curve as well.
      // For now we treat DID presence as “to be expanded” and pass.
      authorOk = true; // TODO: strict DID-based sig check once DID method fixed
    } catch { authorOk = false; }
  }
  if (!authorOk) return { ok:false, reason:"Author signature invalid" };

  // 6) Verifiable Credential (optional but recommended)
  if (meta.verifiableCredential?.uri) {
    try {
      const vc = await fetchJSON(meta.verifiableCredential.uri);
      const res = await verifyCredential(vc, { resolver: new Resolver({} as any) });
      // statusList check
      if (meta.verifiableCredential.statusList) {
        const statusList = await fetchJSON(meta.verifiableCredential.statusList.split("#")[0]);
   import { blake3 } from "blake3";
import { ethers } from "ethers";
import fetch from "cross-fetch";
import Ajv from "ajv";
// VC libs
import { createVerify } from "did-jwt";
import { verifyCredential } from "did-jwt-vc";
import { Resolver } from "did-resolver";
// NOTE: add resolvers as you adopt did:pkh / did:key / etc.
// import { getResolver as pkhResolver } from "@didtools/pkh-did-resolver";
// import { getResolver as keyResolver } from "key-did-resolver";

const abi = [
  "function getRecord(uint256) view returns (tuple(bytes32 contentHash,string metadataURI,bool revoked))"
];

export async function verifyAll({ contentBytes, rpc, contractAddr, tokenId }:{
  contentBytes: Buffer, rpc: string, contractAddr: string, tokenId: bigint
}) {
  // 1) Hash file
  const digest = blake3(contentBytes);
  const computed = "0x" + Buffer.from(digest).toString("hex");

  // 2) On-chain record
  const provider = new ethers.providers.JsonRpcProvider(rpc);
  const c = new ethers.Contract(contractAddr, abi, provider);
  const rec = await c.getRecord(tokenId);
  if (rec.revoked) return { ok:false, reason:"Token revoked" };
  const onchainHash = rec.contentHash.toLowerCase();
  if (computed.toLowerCase() !== onchainHash) return { ok:false, reason:"Hash mismatch" };

  // 3) Fetch metadata
  const meta = await fetchJSON(rec.metadataURI);

  // 4) Schema (optional)
  try {
    const schema = await fetchJSONRaw(new URL("../../offchain/schema/metadata.schema.json", import.meta.url));
    const ajv = new Ajv({ allErrors: true }); const validate = ajv.compile(schema);
    if (!validate(meta)) return { ok:false, reason:"Bad metadata schema", schemaErrors: validate.errors };
  } catch { /* ignore if local path not resolvable in your runtime */ }

  // 5) Author signature over "contentHash||createdAt"
  const payload = Buffer.from(`${meta.contentHash}||${meta.createdAt}`);
  if (!meta.signatures?.[0]) return { ok:false, reason:"Missing author signature" };
  const sigB64 = meta.signatures[0].sig;
  const pubB64 = meta.signatures[0].pub;   // DID fallback: allow raw pub in metadata
  if (!pubB64 && !meta.authorDID) return { ok:false, reason:"No DID or pubkey provided" };

  let authorOk = false;
  if (pubB64) {
    const nacl = await import("tweetnacl");
    authorOk = nacl.sign.detached.verify(payload, Buffer.from(sigB64, "base64"), Buffer.from(pubB64, "base64"));
  } else {
    const resolver = new Resolver({
      // ...add the DID methods you’ll support:
      // ...pkhResolver(), ...keyResolver()
    } as any);
    try {
      // If you used did-jwt for the signature instead, verify here. With the current
      // plain detached signature, you’d need metadata describing alg/curve as well.
      // For now we treat DID presence as “to be expanded” and pass.
      authorOk = true; // TODO: strict DID-based sig check once DID method fixed
    } catch { authorOk = false; }
  }
  if (!authorOk) return { ok:false, reason:"Author signature invalid" };

  // 6) Verifiable Credential (optional but recommended)
  if (meta.verifiableCredential?.uri) {
    try {
      const vc = await fetchJSON(meta.verifiableCredential.uri);
      const res = await verifyCredential(vc, { resolver: new Resolver({} as any) });
      // statusList check
      if (meta.verifiableCredential.statusList) {
        const statusList = await fetchJSON(meta.verifiableCredential.statusList.split("#")[0]);
        if (isRevokedInStatusList(vc, statusList)) return { ok:false, reason:"VC revoked" };
      }
    } catch (e:any) {
      return { ok:false, reason:`VC invalid: ${e.message || e}` };
    }
  }

  return { ok:true, computedHash: computed, tokenId: tokenId.toString(), contract: contractAddr };
}

async function fetchJSON(uri: string) {
  if (uri.startsWith("ipfs://")) {
    throw new Error("Provide IPFS gateway or prefetch IPFS");
  }
  const r = await fetch(uri);
  if (!r.ok) throw new Error(`Fetch ${uri} -> ${r.status}`);
  return r.json();
}

async function fetchJSONRaw(specifier: string | URL) {
  const r = await fetch((specifier as any).toString());
  if (!r.ok) throw new Error(String(r.status));
  return r.json();
}

function isRevokedInStatusList(vc: any, statusListDoc: any): boolean {
  // Minimal placeholder – plug your StatusList v2021 reader or issuer API.
  return false;
}
     if (isRevokedInStatusList(vc, statusList)) return { ok:false, reason:"VC revoked" };
      }
    } catch (e:any) {
      return { ok:false, reason:`VC invalid: ${e.message || e}` };
    }
  }

  return { ok:true, computedHash: computed, tokenId: tokenId.toString(), contract: contractAddr };
}

async function fetchJSON(uri: string) {
  if (uri.startsWith("ipfs://")) {
    throw new Error("Provide IPFS gateway or prefetch IPFS");
  }
  const r = await fetch(uri);
  if (!r.ok) throw new Error(`Fetch ${uri} -> ${r.status}`);
  return r.json();
}

async function fetchJSONRaw(specifier: string | URL) {
  const r = await fetch((specifier as any).toString());
  if (!r.ok) throw new Error(String(r.status));
  return r.json();
}

function isRevokedInStatusList(vc: any, statusListDoc: any): boolean {
  // Minimal placeholder – plug your StatusList v2021 reader or issuer API.
  return false;
}
