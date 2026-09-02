import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
const file = "/Users/xyr/Documents/agent/tau2-bench-v2/outputs/metadata_id_audit/document_name_mapping.xlsx";
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
const check = await wb.inspect({ kind: "table", sheetId: "Document Names", range: "A1:H74", include: "values", tableMaxRows: 5, tableMaxCols: 8, maxChars: 4000 });
console.log(check.ndjson);
const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 50 }, maxChars: 1000 });
console.log(errors.ndjson);
for (const [name, range] of [["Summary", "A1:B18"], ["Document Names", "A1:H20"]]) {
  const blob = await wb.render({ sheetName: name, range, scale: 1, format: "png" });
  await fs.writeFile(`/Users/xyr/Documents/agent/tau2-bench-v2/outputs/metadata_id_audit/${name.replaceAll(" ", "_")}.png`, new Uint8Array(await blob.arrayBuffer()));
}
