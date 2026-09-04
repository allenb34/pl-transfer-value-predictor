// Standalone validation: runs the SAME prediction logic embedded in index.html
// (copy-pasted verbatim below) against docs/model_params.json, for a handful
// of real players, and prints the result so it can be diffed against
// predict.py's output for the same players.
//
// Not loaded by index.html itself - this is a one-off correctness check, run
// with: node docs/validate_js_math.mjs
import { readFileSync } from "node:fs";

const MODEL_PARAMS = JSON.parse(readFileSync(new URL("./model_params.json", import.meta.url)));

// --- verbatim copy of the prediction logic from index_template.html ---
function encodePositionByName(position) {
  const encoded = {};
  for (const cat of MODEL_PARAMS.position_categories) {
    encoded["position_" + cat] = (cat === position) ? 1 : 0;
  }
  return encoded;
}

function buildFeatureVector(goals, assists, age, position) {
  const values = {
    goals: goals,
    assists: assists,
    age: age,
    ...encodePositionByName(position),
  };
  return MODEL_PARAMS.feature_order.map((name) => {
    if (!(name in values)) {
      throw new Error("Missing feature '" + name + "'");
    }
    return values[name];
  });
}

function predictTransferValue(goals, assists, age, position) {
  const x = buildFeatureVector(goals, assists, age, position);
  let logPrediction = MODEL_PARAMS.intercept;
  for (let i = 0; i < x.length; i++) {
    logPrediction += MODEL_PARAMS.coefficients[i] * x[i];
  }
  return Math.expm1(logPrediction);
}
// --- end verbatim copy ---

const testCases = [
  { name: "Erling Haaland", position: "Offence", age: 26, goals: 2, assists: 0, pythonPred: 50097030.44852972 },
  { name: "Richarlison", position: "Offence", age: 29, goals: 0, assists: 0, pythonPred: 16609005.357018903 },
  { name: "Adam Smith", position: "Defence", age: 35, goals: 0, assists: 0, pythonPred: 7600615.576073731 },
];

console.log("name".padEnd(18), "js_prediction".padStart(20), "python_prediction".padStart(20), "abs_diff".padStart(14));
let allMatch = true;
for (const tc of testCases) {
  const jsPred = predictTransferValue(tc.goals, tc.assists, tc.age, tc.position);
  const diff = Math.abs(jsPred - tc.pythonPred);
  const relDiff = diff / tc.pythonPred;
  if (relDiff > 1e-6) allMatch = false;
  console.log(
    tc.name.padEnd(18),
    jsPred.toFixed(4).padStart(20),
    tc.pythonPred.toFixed(4).padStart(20),
    diff.toExponential(3).padStart(14)
  );
}
console.log(allMatch ? "\nPASS: all predictions match Python within float rounding error." : "\nFAIL: mismatch exceeds rounding error.");
