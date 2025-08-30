import fs from "fs";
import dotenv from "dotenv";
import { verifyAll } from "./verify-lib";
dotenv.config();

(async () => {
  const args = require("minimist")(process.argv.slice(2));
  const rpc = args.rpc || process.env.RPC_URL;
  const contract = args.contract;
  const token = BigInt(args.token);
  const file = args.file;
  const buf = fs.readFileSync(file);
  const res = await verifyAll({ contentBytes: buf, rpc, contractAddr: contract, tokenId: token });
  console.log(JSON.stringify(res, null, 2));
  process.exit(res.ok ? 0 : 2);
})();
