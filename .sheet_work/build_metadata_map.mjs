import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "/Users/xyr/Documents/agent/tau2-bench-v2";
const docsDir = path.join(root, "data/tau2/domains/banking_knowledge/documents");
const outputPath = path.join(root, "outputs/metadata_id_audit/document_name_mapping.xlsx");

const files = (await fs.readdir(docsDir)).filter((name) => name.endsWith(".json"));
const counts = new Map();
for (const name of files) {
  const doc = JSON.parse(await fs.readFile(path.join(docsDir, name), "utf8"));
  const stem = doc.id.replace(/_\d+$/, "");
  counts.set(stem, (counts.get(stem) ?? 0) + 1);
}

const prefixes = [
  ["doc_business_checking_accounts_", "business_checking_accounts", "product", "business_checking_account"],
  ["doc_business_credit_cards_", "business_credit_cards", "product", "business_credit_card"],
  ["doc_business_savings_accounts_", "business_savings_accounts", "product", "business_savings_account"],
  ["doc_checking_accounts_", "checking_accounts", "product", "checking_account"],
  ["doc_credit_cards_", "credit_cards", "product", "credit_card"],
  ["doc_savings_accounts_", "savings_accounts", "product", "savings_account"],
  ["doc_buy_now_pay_later_", "buy_now_pay_later", "service", ""],
  ["doc_everyone_pay_", "everyone_pay", "service", ""],
  ["doc_bank_accounts_", "bank_accounts", "topic", ""],
  ["doc_personal_subscriptions_", "personal_subscriptions", "service", ""],
  ["doc_customer_support_", "customer_support", "topic", ""],
];

function classify(stem) {
  const match = prefixes.find(([prefix]) => stem.startsWith(prefix));
  if (!match) return { namespace: "unknown", resourceType: "unknown", category: "", segment: "unknown", candidate: "" };
  const [prefix, namespace, resourceType, category] = match;
  const remainder = stem.slice(prefix.length).replace(/_\(general\)$/, "");
  const isGeneral = stem.endsWith("_(general)");
  const candidate = resourceType === "product" && !isGeneral ? remainder.replaceAll("_", " ") : "";
  return {
    namespace,
    resourceType,
    category,
    segment: namespace.startsWith("business_") ? "business" : "unknown",
    candidate,
    isGeneral,
  };
}

const rows = [...counts.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([stem, count]) => {
  const info = classify(stem);
  return [stem, count, info.namespace, info.resourceType, info.category, info.segment, info.candidate, info.isGeneral];
});

const summaryMap = new Map();
for (const row of rows) summaryMap.set(row[2], (summaryMap.get(row[2]) ?? 0) + 1);
const summary = [...summaryMap.entries()].sort(([a], [b]) => a.localeCompare(b));

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("Summary");
const detailSheet = workbook.worksheets.add("Document Names");
summarySheet.showGridLines = false;
detailSheet.showGridLines = false;

summarySheet.getRange("A1:B1").merge();
summarySheet.getRange("A1").values = [["Banking Knowledge ID Stem Mapping"]];
summarySheet.getRange("A3:B5").values = [
  ["Total documents", files.length],
  ["Unique document names", rows.length],
  ["Parsing source", "ID only (title not used)"],
];
summarySheet.getRange("A7:B7").values = [["Namespace", "Unique names"]];
summarySheet.getRange(`A8:B${7 + summary.length}`).values = summary;

detailSheet.getRange("A1:H1").merge();
detailSheet.getRange("A1").values = [["Unique document names (末尾数字已去除)"]];
detailSheet.getRange("A3:H3").values = [["ID stem", "Document count", "Namespace", "Resource type", "Product category", "Customer segment", "Product name candidate", "Explicit general"]];
detailSheet.getRange(`A4:H${3 + rows.length}`).values = rows;

for (const sheet of [summarySheet, detailSheet]) {
  sheet.getRange("A1").format = { fill: "#16324F", font: { bold: true, color: "#FFFFFF", size: 14 }, horizontalAlignment: "left", verticalAlignment: "center" };
  sheet.getRange("A1").format.rowHeight = 28;
}
summarySheet.getRange("A7:B7").format = { fill: "#D9EAF7", font: { bold: true, color: "#16324F" }, borders: { preset: "bottom", style: "medium", color: "#5B7C99" } };
detailSheet.getRange("A3:H3").format = { fill: "#D9EAF7", font: { bold: true, color: "#16324F" }, wrapText: true, borders: { preset: "bottom", style: "medium", color: "#5B7C99" } };
detailSheet.getRange(`A4:H${3 + rows.length}`).format.borders = { insideHorizontal: { style: "thin", color: "#E1E7ED" } };
summarySheet.getRange("A3:B5").format.borders = { preset: "all", style: "thin", color: "#E1E7ED" };
summarySheet.getRange("B3:B4").format.numberFormat = [["#,##0"], ["#,##0"]];
summarySheet.getRange("A1:D20").format.columnWidth = 24;
summarySheet.getRange("B1:B20").format.columnWidth = 18;
detailSheet.getRange("A:A").format.columnWidth = 52;
detailSheet.getRange("B:B").format.columnWidth = 14;
detailSheet.getRange("C:F").format.columnWidth = 24;
detailSheet.getRange("G:G").format.columnWidth = 32;
detailSheet.getRange("H:H").format.columnWidth = 16;
detailSheet.getRange("A3:H74").format.verticalAlignment = "center";
detailSheet.freezePanes.freezeRows(3);
summarySheet.freezePanes.freezeRows(7);

const out = await SpreadsheetFile.exportXlsx(workbook);
await out.save(outputPath);
console.log(JSON.stringify({ outputPath, documents: files.length, uniqueNames: rows.length, namespaces: summary.length }));
