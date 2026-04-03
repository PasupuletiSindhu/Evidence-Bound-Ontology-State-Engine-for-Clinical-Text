#!/usr/bin/env python3
"""Generate paraphrase and QA evaluation files with configurable sizes.

Default outputs remain backward compatible:
- paraphrase_sets_30.json (30 facts x 10 paraphrases)
- qa_50.json (50 questions)

All facts come from the built-in curated list (real medication / condition style names).
If `--num_facts` is larger than that list, the script uses every curated fact available
and prints a short note to stderr (no synthetic DrugN/ConditionN placeholders).

Example:
  python baselines/data/scripts/generate_30_paraphrase_50_qa.py --num_facts 50 --qa_count 100
"""
import argparse
import json
import re
import sys
from pathlib import Path

# Block legacy synthetic placeholders (Drug37 / Condition36) from ever being written again.
_PLACEHOLDER_DRUG_COND = re.compile(
    r"\b(?:Drug|Condition)\d+\b",
    flags=re.IGNORECASE,
)

PARAPHRASE_SETS_30 = [
    {"fact": "Aspirin causes gastric bleeding.", "paraphrases": [
        "Aspirin may cause gastric bleeding in some patients.", "Gastric bleeding may occur due to Aspirin.",
        "Aspirin is associated with gastric hemorrhage.", "Patients taking Aspirin can develop gastric bleeding.",
        "Aspirin can lead to bleeding in the stomach.", "Gastric hemorrhage is a known adverse effect of Aspirin.",
        "Use of Aspirin may result in gastric bleeding.", "Aspirin has been linked to stomach bleeding.",
        "Gastric bleeding can be caused by Aspirin.", "Aspirin increases the risk of gastric bleeding."]},
    {"fact": "Metformin is used to treat type 2 diabetes.", "paraphrases": [
        "Metformin is used to treat type 2 diabetes.", "Type 2 diabetes can be treated with Metformin.",
        "Metformin is a first-line treatment for type 2 diabetes.", "Patients with type 2 diabetes often take Metformin.",
        "Metformin helps control blood sugar in type 2 diabetes.", "Metformin is prescribed for the treatment of type 2 diabetes.",
        "Type 2 diabetes is commonly treated with Metformin.", "Metformin is an effective drug for type 2 diabetes.",
        "Metformin treats type 2 diabetes mellitus.", "For type 2 diabetes, Metformin is frequently used."]},
    {"fact": "Warfarin increases the risk of bleeding.", "paraphrases": [
        "Warfarin increases the risk of bleeding.", "Bleeding risk is elevated with Warfarin use.",
        "Warfarin can cause bleeding complications.", "Patients on Warfarin have an increased bleeding risk.",
        "Warfarin is associated with a higher risk of bleeding.", "Bleeding may occur in patients taking Warfarin.",
        "Warfarin therapy increases the chance of bleeding.", "The use of Warfarin raises the risk of bleeding.",
        "Warfarin has been linked to bleeding events.", "Bleeding is a known adverse effect of Warfarin."]},
    {"fact": "Ibuprofen can reduce fever and pain.", "paraphrases": [
        "Ibuprofen can reduce fever and pain.", "Ibuprofen is used to lower fever and relieve pain.",
        "Fever and pain may be reduced by Ibuprofen.", "Ibuprofen helps reduce fever and alleviate pain.",
        "Ibuprofen is effective for fever and pain relief.", "Pain and fever can be treated with Ibuprofen.",
        "Ibuprofen reduces fever and pain in patients.", "Ibuprofen is taken to reduce fever and pain.",
        "Fever and pain relief are effects of Ibuprofen.", "Ibuprofen can bring down fever and reduce pain."]},
    {"fact": "Acetaminophen is metabolized by the liver.", "paraphrases": [
        "Acetaminophen is metabolized by the liver.", "The liver metabolizes Acetaminophen.",
        "Acetaminophen metabolism occurs in the liver.", "Liver enzymes metabolize Acetaminophen.",
        "Acetaminophen is broken down by the liver.", "Metabolism of Acetaminophen takes place in the liver.",
        "The liver is responsible for Acetaminophen metabolism.", "Acetaminophen is hepatically metabolized.",
        "Liver function affects Acetaminophen metabolism.", "Acetaminophen undergoes hepatic metabolism."]},
]

# Add more facts (same structure: fact + 10 paraphrases). Reuse patterns for brevity.
EXTRA_FACTS = [
    ("Statins lower cholesterol.", ["Statins are used to lower cholesterol.", "Cholesterol can be reduced by statins.", "Statins reduce cholesterol levels.", "Lowering cholesterol is an effect of statins.", "Statins help lower LDL cholesterol.", "Cholesterol-lowering drugs include statins.", "Statins decrease cholesterol in the blood.", "Taking statins can lower your cholesterol.", "Statins are prescribed to reduce cholesterol.", "Cholesterol is lowered by statin therapy."]),
    ("Insulin regulates blood glucose.", ["Insulin regulates blood glucose levels.", "Blood glucose is regulated by insulin.", "Insulin helps control blood sugar.", "Glucose regulation depends on insulin.", "Insulin is involved in glucose regulation.", "Blood sugar is regulated by insulin.", "Insulin controls glucose in the bloodstream.", "Glucose levels are regulated by insulin.", "Insulin plays a role in blood glucose regulation.", "Regulation of blood glucose involves insulin."]),
    ("Penicillin treats bacterial infections.", ["Penicillin is used to treat bacterial infections.", "Bacterial infections can be treated with penicillin.", "Penicillin treats infections caused by bacteria.", "Penicillin is an antibiotic for bacterial infections.", "Bacterial infections are treated with penicillin.", "Penicillin is effective against bacterial infections.", "Treating bacterial infections is a use of penicillin.", "Penicillin helps fight bacterial infections.", "Bacterial infections may be treated with penicillin.", "Penicillin is prescribed for bacterial infections."]),
    ("Morphine relieves severe pain.", ["Morphine relieves severe pain.", "Severe pain can be relieved by morphine.", "Morphine is used for severe pain relief.", "Morphine provides relief from severe pain.", "Severe pain is often treated with morphine.", "Morphine is a strong pain reliever.", "Pain relief from morphine is effective for severe pain.", "Morphine helps relieve severe pain.", "Severe pain may be managed with morphine.", "Morphine is given to relieve severe pain."]),
    ("Beta blockers reduce blood pressure.", ["Beta blockers reduce blood pressure.", "Blood pressure can be lowered by beta blockers.", "Beta blockers are used to reduce blood pressure.", "Lowering blood pressure is an effect of beta blockers.", "Beta blockers help lower blood pressure.", "Blood pressure is reduced with beta blocker therapy.", "Beta blockers decrease blood pressure.", "Taking beta blockers can reduce blood pressure.", "Beta blockers are prescribed for high blood pressure.", "Blood pressure reduction is achieved with beta blockers."]),
    ("Anticoagulants prevent blood clots.", ["Anticoagulants prevent blood clots.", "Blood clots can be prevented by anticoagulants.", "Anticoagulants are used to prevent clotting.", "Prevention of blood clots is a use of anticoagulants.", "Anticoagulants help prevent blood clots.", "Blood clot prevention is an effect of anticoagulants.", "Anticoagulants reduce the risk of blood clots.", "Taking anticoagulants helps prevent clots.", "Anticoagulants are prescribed to prevent blood clots.", "Blood clots are less likely with anticoagulant therapy."]),
    ("Corticosteroids reduce inflammation.", ["Corticosteroids reduce inflammation.", "Inflammation can be reduced by corticosteroids.", "Corticosteroids are used to reduce inflammation.", "Reducing inflammation is an effect of corticosteroids.", "Corticosteroids help decrease inflammation.", "Inflammation is reduced with corticosteroid treatment.", "Corticosteroids decrease inflammation.", "Corticosteroids are anti-inflammatory.", "Inflammation may be controlled with corticosteroids.", "Corticosteroids suppress inflammation."]),
    ("ACE inhibitors treat hypertension.", ["ACE inhibitors treat hypertension.", "Hypertension can be treated with ACE inhibitors.", "ACE inhibitors are used for hypertension.", "ACE inhibitors help treat high blood pressure.", "Hypertension is treated with ACE inhibitor therapy.", "ACE inhibitors are prescribed for hypertension.", "Treating hypertension is a use of ACE inhibitors.", "ACE inhibitors lower blood pressure in hypertensive patients.", "Hypertension may be managed with ACE inhibitors.", "ACE inhibitors are effective for hypertension."]),
    ("Metformin can cause lactic acidosis.", ["Metformin can cause lactic acidosis.", "Lactic acidosis may occur with metformin use.", "Metformin is associated with lactic acidosis.", "Lactic acidosis is a risk of metformin.", "Metformin use can lead to lactic acidosis.", "Lactic acidosis has been linked to metformin.", "Metformin may cause lactic acidosis in some patients.", "A rare side effect of metformin is lactic acidosis.", "Lactic acidosis can be caused by metformin.", "Metformin carries a risk of lactic acidosis."]),
    ("Warfarin interacts with vitamin K.", ["Warfarin interacts with vitamin K.", "Vitamin K can interact with warfarin.", "Warfarin and vitamin K interact.", "There is an interaction between warfarin and vitamin K.", "Vitamin K affects warfarin activity.", "Warfarin efficacy is influenced by vitamin K.", "Vitamin K intake can alter warfarin effect.", "Warfarin interacts with vitamin K in the body.", "Patients on warfarin should consider vitamin K.", "Vitamin K and warfarin have a known interaction."]),
    ("Aspirin prevents heart attacks.", ["Aspirin prevents heart attacks.", "Heart attacks can be prevented by aspirin.", "Aspirin is used to prevent heart attacks.", "Prevention of heart attack is a use of aspirin.", "Aspirin helps prevent heart attacks.", "Low-dose aspirin may reduce heart attack risk.", "Aspirin reduces the risk of heart attack.", "Heart attack prevention is an effect of aspirin.", "Aspirin is taken to prevent heart attacks.", "Aspirin therapy can prevent heart attacks."]),
    ("Omeprazole reduces stomach acid.", ["Omeprazole reduces stomach acid.", "Stomach acid can be reduced by omeprazole.", "Omeprazole is used to reduce stomach acid.", "Omeprazole decreases gastric acid production.", "Reducing stomach acid is an effect of omeprazole.", "Omeprazole helps lower stomach acid.", "Stomach acid is reduced with omeprazole.", "Omeprazole is a proton pump inhibitor that reduces acid.", "Acid reflux may be treated with omeprazole.", "Omeprazole suppresses stomach acid."]),
    ("Lithium treats bipolar disorder.", ["Lithium treats bipolar disorder.", "Bipolar disorder can be treated with lithium.", "Lithium is used to treat bipolar disorder.", "Lithium is a mood stabilizer for bipolar disorder.", "Bipolar disorder is treated with lithium.", "Lithium helps manage bipolar disorder.", "Lithium is prescribed for bipolar disorder.", "Treating bipolar disorder is a use of lithium.", "Bipolar patients may take lithium.", "Lithium is effective for bipolar disorder."]),
    ("Levothyroxine treats hypothyroidism.", ["Levothyroxine treats hypothyroidism.", "Hypothyroidism can be treated with levothyroxine.", "Levothyroxine is used for hypothyroidism.", "Levothyroxine replaces thyroid hormone.", "Hypothyroidism is treated with levothyroxine.", "Levothyroxine is the standard treatment for hypothyroidism.", "Patients with hypothyroidism take levothyroxine.", "Levothyroxine helps treat underactive thyroid.", "Hypothyroidism may be managed with levothyroxine.", "Levothyroxine is prescribed for hypothyroidism."]),
    ("Albuterol relieves asthma symptoms.", ["Albuterol relieves asthma symptoms.", "Asthma symptoms can be relieved by albuterol.", "Albuterol is used to relieve asthma symptoms.", "Albuterol is a rescue inhaler for asthma.", "Asthma symptoms are relieved with albuterol.", "Albuterol helps open airways in asthma.", "Albuterol is used for acute asthma relief.", "Relief of asthma symptoms is an effect of albuterol.", "Asthma patients use albuterol for symptom relief.", "Albuterol quickly relieves asthma symptoms."]),
    ("Furosemide is a diuretic.", ["Furosemide is a diuretic.", "Furosemide is used as a diuretic.", "Furosemide causes increased urine output.", "Diuretic effect is a property of furosemide.", "Furosemide is a loop diuretic.", "Furosemide helps remove excess fluid.", "As a diuretic, furosemide increases urination.", "Furosemide is prescribed as a diuretic.", "Furosemide has diuretic action.", "The diuretic furosemide is used for fluid overload."]),
    ("Digoxin increases heart contractility.", ["Digoxin increases heart contractility.", "Heart contractility is increased by digoxin.", "Digoxin is used to increase heart contractility.", "Digoxin strengthens heart contractions.", "Increased contractility is an effect of digoxin.", "Digoxin improves the heart's pumping ability.", "Heart contractility may be improved with digoxin.", "Digoxin is a positive inotrope.", "Digoxin increases the force of heart contractions.", "Contractility of the heart is increased by digoxin."]),
    ("Prednisone suppresses the immune system.", ["Prednisone suppresses the immune system.", "The immune system can be suppressed by prednisone.", "Prednisone is an immunosuppressant.", "Immunosuppression is an effect of prednisone.", "Prednisone reduces immune activity.", "Prednisone is used to suppress immunity.", "Immune suppression occurs with prednisone use.", "Prednisone dampens the immune response.", "Patients on prednisone may have suppressed immunity.", "Prednisone has immunosuppressive effects."]),
    ("Fluoxetine treats depression.", ["Fluoxetine treats depression.", "Depression can be treated with fluoxetine.", "Fluoxetine is used to treat depression.", "Fluoxetine is an antidepressant.", "Depression is treated with fluoxetine.", "Fluoxetine helps alleviate depression.", "Fluoxetine is prescribed for depression.", "Treating depression is a use of fluoxetine.", "Depression may be managed with fluoxetine.", "Fluoxetine is effective for depression."]),
    ("Amoxicillin treats ear infections.", ["Amoxicillin treats ear infections.", "Ear infections can be treated with amoxicillin.", "Amoxicillin is used for ear infections.", "Amoxicillin is commonly prescribed for ear infections.", "Ear infections are often treated with amoxicillin.", "Amoxicillin helps clear ear infections.", "Otitis media may be treated with amoxicillin.", "Amoxicillin is an antibiotic for ear infections.", "Ear infection treatment often includes amoxicillin.", "Amoxicillin is effective against ear infections."]),
    ("Atorvastatin lowers LDL cholesterol.", ["Atorvastatin lowers LDL cholesterol.", "LDL cholesterol can be lowered by atorvastatin.", "Atorvastatin is used to lower LDL cholesterol.", "Atorvastatin reduces LDL levels.", "LDL cholesterol is reduced with atorvastatin.", "Lowering LDL is an effect of atorvastatin.", "Atorvastatin helps reduce bad cholesterol.", "Atorvastatin is a statin that lowers LDL.", "LDL cholesterol may be reduced with atorvastatin.", "Atorvastatin decreases LDL cholesterol."]),
    ("Sertraline treats anxiety.", ["Sertraline treats anxiety.", "Anxiety can be treated with sertraline.", "Sertraline is used to treat anxiety.", "Sertraline is prescribed for anxiety disorders.", "Anxiety is often treated with sertraline.", "Sertraline helps reduce anxiety.", "Sertraline is an SSRI used for anxiety.", "Anxiety disorders may be managed with sertraline.", "Treating anxiety is a use of sertraline.", "Sertraline is effective for anxiety."]),
    ("Hydrochlorothiazide is a diuretic.", ["Hydrochlorothiazide is a diuretic.", "Hydrochlorothiazide is used as a diuretic.", "Hydrochlorothiazide increases urine output.", "As a diuretic, hydrochlorothiazide removes fluid.", "Hydrochlorothiazide is a thiazide diuretic.", "Hydrochlorothiazide helps lower blood pressure by diuresis.", "Diuretic effect is a property of hydrochlorothiazide.", "Hydrochlorothiazide is prescribed for fluid retention.", "Hydrochlorothiazide has diuretic action.", "The diuretic hydrochlorothiazide is commonly used."]),
    ("Gabapentin treats nerve pain.", ["Gabapentin treats nerve pain.", "Nerve pain can be treated with gabapentin.", "Gabapentin is used for neuropathic pain.", "Gabapentin helps relieve nerve pain.", "Nerve pain is often treated with gabapentin.", "Gabapentin is prescribed for nerve pain.", "Neuropathic pain may be managed with gabapentin.", "Gabapentin is effective for nerve pain.", "Treating nerve pain is a use of gabapentin.", "Gabapentin reduces nerve pain."]),
    ("Losartan treats high blood pressure.", ["Losartan treats high blood pressure.", "High blood pressure can be treated with losartan.", "Losartan is used for hypertension.", "Losartan is an ARB for blood pressure.", "Hypertension is treated with losartan.", "Losartan helps lower blood pressure.", "Losartan is prescribed for high blood pressure.", "Blood pressure may be controlled with losartan.", "Losartan is effective for hypertension.", "Treating high blood pressure is a use of losartan."]),
    ("Metronidazole treats bacterial infections.", ["Metronidazole treats bacterial infections.", "Bacterial infections can be treated with metronidazole.", "Metronidazole is an antibiotic for certain bacteria.", "Metronidazole is used for anaerobic infections.", "Bacterial infections are treated with metronidazole.", "Metronidazole is effective against anaerobic bacteria.", "Metronidazole helps fight bacterial infections.", "Certain bacterial infections may be treated with metronidazole.", "Metronidazole is prescribed for bacterial infections.", "Treating bacterial infections is a use of metronidazole."]),
    ("Sumatriptan treats migraine headaches.", ["Sumatriptan treats migraine headaches.", "Migraine headaches can be treated with sumatriptan.", "Sumatriptan is used for acute migraine.", "Sumatriptan relieves migraine attacks.", "Migraine is often treated with sumatriptan.", "Sumatriptan is a triptan for migraine.", "Sumatriptan helps stop migraine headaches.", "Patients with migraine may take sumatriptan.", "Sumatriptan is prescribed for migraine.", "Treating migraine is a use of sumatriptan."]),
    ("Clopidogrel prevents blood clots.", ["Clopidogrel prevents blood clots.", "Blood clots can be prevented with clopidogrel.", "Clopidogrel is an antiplatelet drug.", "Clopidogrel reduces clot formation.", "Prevention of clots is a use of clopidogrel.", "Clopidogrel helps prevent arterial thrombosis.", "Clopidogrel is used to prevent clots after stents.", "Clopidogrel lowers the risk of clotting.", "Patients at clot risk may take clopidogrel.", "Clopidogrel is prescribed to prevent blood clots."]),
    ("Montelukast treats asthma.", ["Montelukast treats asthma.", "Asthma can be treated with montelukast.", "Montelukast is used for asthma control.", "Montelukast is a leukotriene modifier for asthma.", "Asthma patients may take montelukast.", "Montelukast helps manage asthma.", "Montelukast is prescribed for asthma.", "Treating asthma is a use of montelukast.", "Montelukast improves asthma symptoms.", "Asthma maintenance may include montelukast."]),
    ("Allopurinol treats gout.", ["Allopurinol treats gout.", "Gout can be treated with allopurinol.", "Allopurinol lowers uric acid in gout.", "Allopurinol is used for chronic gout.", "Gout flares are prevented with allopurinol long-term.", "Allopurinol is prescribed for hyperuricemia in gout.", "Patients with gout may take allopurinol.", "Allopurinol helps prevent gout attacks.", "Treating gout is a use of allopurinol.", "Gout management often includes allopurinol."]),
    ("Carbamazepine treats epilepsy.", ["Carbamazepine treats epilepsy.", "Epilepsy can be treated with carbamazepine.", "Carbamazepine is an anticonvulsant.", "Carbamazepine is used for seizure control.", "Epilepsy patients may take carbamazepine.", "Carbamazepine helps reduce seizures.", "Carbamazepine is prescribed for epilepsy.", "Treating epilepsy is a use of carbamazepine.", "Seizure disorders may be managed with carbamazepine.", "Carbamazepine is effective for some epilepsies."]),
    ("Finasteride treats benign prostatic hyperplasia.", ["Finasteride treats benign prostatic hyperplasia.", "BPH can be treated with finasteride.", "Finasteride shrinks the prostate in BPH.", "Finasteride is used for enlarged prostate.", "Benign prostatic hyperplasia is treated with finasteride.", "Finasteride improves urinary symptoms in BPH.", "Patients with BPH may take finasteride.", "Finasteride is prescribed for BPH.", "Treating BPH is a use of finasteride.", "Finasteride reduces prostate volume over time."]),
    ("Azithromycin treats pneumonia.", ["Azithromycin treats pneumonia.", "Pneumonia can be treated with azithromycin.", "Azithromycin is a macrolide for pneumonia.", "Azithromycin is used for community-acquired pneumonia.", "Pneumonia may be treated with azithromycin.", "Azithromycin helps clear bacterial pneumonia.", "Azithromycin is prescribed for pneumonia.", "Treating pneumonia is a use of azithromycin.", "Some pneumonia cases receive azithromycin.", "Azithromycin is effective for certain pneumonias."]),
    ("Isosorbide mononitrate treats angina.", ["Isosorbide mononitrate treats angina.", "Angina can be treated with isosorbide mononitrate.", "Isosorbide mononitrate is a nitrate for angina.", "Isosorbide mononitrate prevents angina episodes.", "Stable angina is treated with isosorbide mononitrate.", "Isosorbide mononitrate dilates coronary vessels.", "Patients with angina may take isosorbide mononitrate.", "Isosorbide mononitrate is prescribed for angina.", "Treating angina is a use of isosorbide mononitrate.", "Angina prophylaxis may include isosorbide mononitrate."]),
    ("Spironolactone treats heart failure.", ["Spironolactone treats heart failure.", "Heart failure can be treated with spironolactone.", "Spironolactone is an aldosterone antagonist in HFrEF.", "Spironolactone reduces mortality in some heart failure.", "Spironolactone is used for congestive heart failure.", "Heart failure patients may take spironolactone.", "Spironolactone helps manage fluid in heart failure.", "Spironolactone is prescribed for heart failure.", "Treating heart failure is a use of spironolactone.", "Spironolactone improves outcomes in selected heart failure."]),
    ("Methotrexate treats rheumatoid arthritis.", ["Methotrexate treats rheumatoid arthritis.", "Rheumatoid arthritis can be treated with methotrexate.", "Methotrexate is a DMARD for RA.", "Methotrexate is a first-line drug for rheumatoid arthritis.", "RA is often treated with methotrexate.", "Methotrexate slows joint damage in RA.", "Patients with RA may take methotrexate.", "Methotrexate is prescribed for rheumatoid arthritis.", "Treating RA is a use of methotrexate.", "Methotrexate reduces inflammation in rheumatoid arthritis."]),
    ("Nitrofurantoin treats urinary tract infections.", ["Nitrofurantoin treats urinary tract infections.", "Urinary tract infections can be treated with nitrofurantoin.", "Nitrofurantoin is used for uncomplicated UTIs.", "Nitrofurantoin concentrates in urine.", "UTIs are sometimes treated with nitrofurantoin.", "Nitrofurantoin is an antibiotic for bladder infections.", "Nitrofurantoin is prescribed for UTIs.", "Treating UTIs is a use of nitrofurantoin.", "Simple cystitis may be treated with nitrofurantoin.", "Nitrofurantoin is effective for many UTIs."]),
    ("Haloperidol treats schizophrenia.", ["Haloperidol treats schizophrenia.", "Schizophrenia can be treated with haloperidol.", "Haloperidol is an antipsychotic.", "Haloperidol is used for psychosis in schizophrenia.", "Schizophrenia patients may receive haloperidol.", "Haloperidol reduces psychotic symptoms.", "Haloperidol is prescribed for schizophrenia.", "Treating schizophrenia is a use of haloperidol.", "Acute agitation in schizophrenia may use haloperidol.", "Haloperidol has long been used for schizophrenia."]),
    ("Phenytoin treats seizures.", ["Phenytoin treats seizures.", "Seizures can be treated with phenytoin.", "Phenytoin is an antiepileptic drug.", "Phenytoin is used for tonic-clonic seizures.", "Seizure disorders may be treated with phenytoin.", "Phenytoin helps control seizures.", "Phenytoin is prescribed for epilepsy seizures.", "Treating seizures is a use of phenytoin.", "Some epilepsy patients take phenytoin.", "Phenytoin stabilizes neuronal firing in seizures."]),
    ("Doxycycline treats Lyme disease.", ["Doxycycline treats Lyme disease.", "Lyme disease can be treated with doxycycline.", "Doxycycline is first-line for early Lyme.", "Doxycycline is used for Borrelia infection.", "Lyme disease often receives doxycycline.", "Doxycycline is an antibiotic for Lyme.", "Doxycycline is prescribed for Lyme disease.", "Treating Lyme disease is a use of doxycycline.", "Early Lyme may be cured with doxycycline.", "Doxycycline covers tick-borne Lyme in many cases."]),
    ("Esomeprazole treats gastroesophageal reflux disease.", ["Esomeprazole treats gastroesophageal reflux disease.", "GERD can be treated with esomeprazole.", "Esomeprazole is a PPI for acid reflux.", "Esomeprazole reduces stomach acid in GERD.", "GERD patients may take esomeprazole.", "Esomeprazole heals esophagitis from reflux.", "Esomeprazole is prescribed for GERD.", "Treating GERD is a use of esomeprazole.", "Heartburn from GERD may improve with esomeprazole.", "Esomeprazole suppresses acid in GERD."]),
    ("Tamsulosin treats benign prostatic hyperplasia.", ["Tamsulosin treats benign prostatic hyperplasia.", "BPH can be treated with tamsulosin.", "Tamsulosin relaxes the prostate smooth muscle.", "Tamsulosin improves urine flow in BPH.", "Tamsulosin is an alpha-blocker for BPH.", "Patients with BPH may take tamsulosin.", "Tamsulosin is prescribed for enlarged prostate symptoms.", "Treating BPH is a use of tamsulosin.", "Urinary symptoms of BPH may improve with tamsulosin.", "Tamsulosin is commonly used for BPH."]),
    ("Rivaroxaban prevents stroke in atrial fibrillation.", ["Rivaroxaban prevents stroke in atrial fibrillation.", "Stroke risk in AF can be reduced with rivaroxaban.", "Rivaroxaban is an anticoagulant for AF.", "Rivaroxaban is used for stroke prevention in atrial fibrillation.", "Atrial fibrillation may be treated with rivaroxaban for stroke risk.", "Rivaroxaban lowers stroke risk in nonvalvular AF.", "Patients with AF may take rivaroxaban.", "Rivaroxaban is prescribed to prevent AF-related stroke.", "Preventing stroke in AF is a use of rivaroxaban.", "Rivaroxaban is an alternative to warfarin in some AF patients."]),
    ("Clindamycin treats skin infections.", ["Clindamycin treats skin infections.", "Skin infections can be treated with clindamycin.", "Clindamycin is used for cellulitis and abscesses.", "Clindamycin is an antibiotic for skin and soft tissue.", "Bacterial skin infections may receive clindamycin.", "Clindamycin is prescribed for skin infections.", "Treating skin infections is a use of clindamycin.", "Some MRSA skin infections use clindamycin.", "Clindamycin helps clear bacterial skin infections.", "Clindamycin is effective for certain skin infections."]),
    ("Bisoprolol treats heart failure.", ["Bisoprolol treats heart failure.", "Heart failure can be treated with bisoprolol.", "Bisoprolol is a beta blocker used in HFrEF.", "Bisoprolol improves survival in some heart failure.", "Heart failure patients may take bisoprolol.", "Bisoprolol is prescribed for chronic heart failure.", "Treating heart failure is a use of bisoprolol.", "Bisoprolol reduces heart rate in heart failure.", "Guideline-directed therapy may include bisoprolol.", "Bisoprolol helps the failing heart over time."]),
    ("Terbinafine treats fungal nail infections.", ["Terbinafine treats fungal nail infections.", "Fungal nail infections can be treated with terbinafine.", "Terbinafine is an oral antifungal for onychomycosis.", "Terbinafine is used for toenail fungus.", "Onychomycosis is treated with terbinafine.", "Terbinafine is prescribed for nail fungus.", "Treating fungal nails is a use of terbinafine.", "Dermatophyte nail infections may use terbinafine.", "Terbinafine clears some fungal nail infections.", "Terbinafine is commonly used for onychomycosis."]),
    ("Prazosin treats nightmares in PTSD.", ["Prazosin treats nightmares in PTSD.", "PTSD nightmares can be treated with prazosin.", "Prazosin is used off-label for trauma-related nightmares.", "Prazosin may reduce nightmare frequency in PTSD.", "Patients with PTSD may take prazosin for sleep.", "Prazosin is an alpha blocker for PTSD nightmares.", "Treating PTSD nightmares is a use of prazosin in some patients.", "Prazosin is prescribed for disturbing dreams in PTSD.", "Sleep disturbance in PTSD may improve with prazosin.", "Prazosin blocks adrenergic effects linked to nightmares."]),
    ("Colchicine treats pericarditis.", ["Colchicine treats pericarditis.", "Pericarditis can be treated with colchicine.", "Colchicine is used with NSAIDs for pericarditis.", "Colchicine reduces recurrence of pericarditis.", "Pericarditis patients may take colchicine.", "Colchicine is prescribed for acute pericarditis.", "Treating pericarditis is a use of colchicine.", "Colchicine is anti-inflammatory in pericarditis.", "Recurrent pericarditis prevention may include colchicine.", "Colchicine is standard adjunct therapy in many pericarditis cases."]),
]

for item in EXTRA_FACTS:
    PARAPHRASE_SETS_30.append({"fact": item[0], "paraphrases": item[1]})

RELATION_ALIASES = {
    "causes": ["causes", "cause", "LABEL_0", "related_to", "associated_with"],
    "treats": ["treats", "treat", "used_to_treat", "LABEL_1"],
    "interacts_with": ["interacts_with", "interacts with", "interaction"],
    "prevents": ["prevents", "prevent", "prevention"],
    "reduces": ["reduces", "reduce", "lower", "lowers"],
    "increases": ["increases", "increase", "raises"],
    "metabolized_by": ["metabolized_by", "metabolized by", "is metabolized by", "metabolism"],
}

def build_qa_50():
    """50 QA examples across core facts."""
    examples = []
    # Fact 1: Aspirin / gastric bleeding (10)
    for q, subj, rel, obj, ans in [
        ("What does Aspirin cause?", "Aspirin", "causes", "?", "gastric bleeding"),
        ("What can cause gastric bleeding?", "?", "causes", "gastric bleeding", "Aspirin"),
        ("Which drug causes gastric hemorrhage?", "?", "causes", "gastric hemorrhage", "Aspirin"),
        ("What adverse effect does Aspirin have?", "Aspirin", "causes", "?", "gastric bleeding"),
        ("What is a side effect of Aspirin?", "Aspirin", "causes", "?", "gastric bleeding"),
        ("Name a drug that causes gastric bleeding.", "?", "causes", "gastric bleeding", "Aspirin"),
        ("Aspirin is associated with what condition?", "Aspirin", "causes", "?", "gastric bleeding"),
        ("What medication can lead to stomach bleeding?", "?", "causes", "stomach bleeding", "Aspirin"),
        ("What does Aspirin increase the risk of?", "Aspirin", "causes", "?", "gastric bleeding"),
        ("Gastric bleeding may be caused by what drug?", "?", "causes", "gastric bleeding", "Aspirin"),
    ]:
        examples.append({"question": q, "subject": subj, "relation": rel, "object": obj, "answer": ans})

    # Fact 2: Metformin / type 2 diabetes (10)
    for q, subj, rel, obj, ans in [
        ("What is Metformin used to treat?", "Metformin", "treats", "?", "type 2 diabetes"),
        ("What drug treats type 2 diabetes?", "?", "treats", "type 2 diabetes", "Metformin"),
        ("Metformin treats what condition?", "Metformin", "treats", "?", "type 2 diabetes"),
        ("Type 2 diabetes is treated with what medication?", "?", "treats", "type 2 diabetes", "Metformin"),
        ("What does Metformin treat?", "Metformin", "treats", "?", "type 2 diabetes"),
        ("Name a drug for type 2 diabetes.", "?", "treats", "type 2 diabetes", "Metformin"),
        ("What condition does Metformin help?", "Metformin", "treats", "?", "type 2 diabetes"),
        ("What medication is used for type 2 diabetes?", "?", "treats", "type 2 diabetes", "Metformin"),
        ("Metformin is prescribed for what?", "Metformin", "treats", "?", "type 2 diabetes"),
        ("What drug is first-line for type 2 diabetes?", "?", "treats", "type 2 diabetes", "Metformin"),
    ]:
        examples.append({"question": q, "subject": subj, "relation": rel, "object": obj, "answer": ans})

    # Fact 3: Warfarin / bleeding (10)
    for q, subj, rel, obj, ans in [
        ("What does Warfarin increase the risk of?", "Warfarin", "causes", "?", "bleeding"),
        ("What drug increases bleeding risk?", "?", "causes", "bleeding", "Warfarin"),
        ("Warfarin is associated with what risk?", "Warfarin", "causes", "?", "bleeding"),
        ("Bleeding risk is elevated with what medication?", "?", "causes", "bleeding", "Warfarin"),
        ("What adverse effect does Warfarin have?", "Warfarin", "causes", "?", "bleeding"),
        ("Name an anticoagulant that increases bleeding risk.", "?", "causes", "bleeding", "Warfarin"),
        ("What can Warfarin cause?", "Warfarin", "causes", "?", "bleeding"),
        ("What medication raises the risk of bleeding?", "?", "causes", "bleeding", "Warfarin"),
        ("Warfarin therapy may lead to what?", "Warfarin", "causes", "?", "bleeding"),
        ("What is a complication of Warfarin?", "Warfarin", "causes", "?", "bleeding"),
    ]:
        examples.append({"question": q, "subject": subj, "relation": rel, "object": obj, "answer": ans})

    # Fact 4: Ibuprofen / fever and pain (10)
    for q, subj, rel, obj, ans in [
        ("What does Ibuprofen reduce?", "Ibuprofen", "treats", "?", "fever and pain"),
        ("What drug reduces fever and pain?", "?", "treats", "fever and pain", "Ibuprofen"),
        ("Ibuprofen is used for what?", "Ibuprofen", "treats", "?", "fever and pain"),
        ("Fever and pain can be treated with what?", "?", "treats", "fever and pain", "Ibuprofen"),
        ("What does Ibuprofen relieve?", "Ibuprofen", "treats", "?", "fever and pain"),
        ("Name a drug for fever and pain.", "?", "treats", "fever and pain", "Ibuprofen"),
        ("What is Ibuprofen effective for?", "Ibuprofen", "treats", "?", "fever and pain"),
        ("What medication reduces fever?", "?", "treats", "fever and pain", "Ibuprofen"),
        ("Ibuprofen helps with what symptoms?", "Ibuprofen", "treats", "?", "fever and pain"),
        ("Pain and fever are relieved by what drug?", "?", "treats", "fever and pain", "Ibuprofen"),
    ]:
        examples.append({"question": q, "subject": subj, "relation": rel, "object": obj, "answer": ans})

    # Fact 5: Acetaminophen / liver (10)
    for q, subj, rel, obj, ans in [
        ("What organ metabolizes Acetaminophen?", "Acetaminophen", "metabolized_by", "?", "liver"),
        ("Acetaminophen is metabolized by what?", "Acetaminophen", "metabolized_by", "?", "liver"),
        ("Where is Acetaminophen metabolized?", "Acetaminophen", "metabolized_by", "?", "liver"),
        ("What drug is metabolized by the liver?", "?", "metabolized_by", "liver", "Acetaminophen"),
        ("Acetaminophen is broken down by what organ?", "Acetaminophen", "metabolized_by", "?", "liver"),
        ("The liver metabolizes what medication?", "?", "metabolized_by", "liver", "Acetaminophen"),
        ("What is Acetaminophen metabolism dependent on?", "Acetaminophen", "metabolized_by", "?", "liver"),
        ("Hepatic metabolism applies to what drug?", "?", "metabolized_by", "liver", "Acetaminophen"),
        ("Acetaminophen undergoes metabolism where?", "Acetaminophen", "metabolized_by", "?", "liver"),
        ("What pain reliever is hepatically metabolized?", "?", "metabolized_by", "liver", "Acetaminophen"),
    ]:
        examples.append({"question": q, "subject": subj, "relation": rel, "object": obj, "answer": ans})

    return examples


def _relation_from_fact(fact_text: str) -> str:
    low = fact_text.lower()
    if "treat" in low:
        return "treats"
    if "prevent" in low:
        return "prevents"
    if "interact" in low:
        return "interacts_with"
    if "metabol" in low:
        return "metabolized_by"
    if "increase" in low or "risk of" in low:
        return "increases"
    if "reduce" in low or "lower" in low:
        return "reduces"
    return "causes"


def _parse_subject_object_from_fact(fact_text: str):
    text = fact_text.strip().rstrip(".")
    # simple parser for "<subject> <verb/phrase> <object>"
    for sep in [
        " treats ",
        " causes ",
        " prevents ",
        " reduces ",
        " lowers ",
        " increases ",
        " interacts with ",
        " is metabolized by ",
    ]:
        if sep in text:
            left, right = text.split(sep, 1)
            return left.strip(), right.strip()
    parts = text.split(" ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return text, "unknown"


def build_qa_from_facts(facts, qa_count: int):
    """Create QA examples from facts with light lexical variety."""
    examples = []
    for item in facts:
        fact = item["fact"]
        subj, obj = _parse_subject_object_from_fact(fact)
        rel = _relation_from_fact(fact)
        examples.append({
            "question": f"What does {subj} {rel.replace('_', ' ')}?",
            "subject": subj,
            "relation": rel,
            "object": "?",
            "answer": obj,
        })
        examples.append({
            "question": f"What {'' if rel.endswith('s') else 'does '}treats {obj}?" if rel == "treats" else f"What is linked to {obj} via {rel.replace('_', ' ')}?",
            "subject": "?",
            "relation": rel,
            "object": obj,
            "answer": subj,
        })
        if len(examples) >= qa_count:
            break
    return examples[:qa_count]


def _curated_paraphrase_sets():
    """Hand-written clinical-style facts only (no DrugN/ConditionN padding)."""
    base = list(PARAPHRASE_SETS_30[:5])
    for item in EXTRA_FACTS:
        base.append({"fact": item[0], "paraphrases": item[1]})
    return base


def build_paraphrase_sets(num_facts: int):
    base = _curated_paraphrase_sets()
    if num_facts > len(base):
        print(
            f"note: num_facts={num_facts} > {len(base)} curated facts; writing {len(base)} sets.",
            file=sys.stderr,
        )
    return base[:num_facts]


def validate_no_placeholder_entities(paraphrase_sets: list) -> None:
    """Fail fast if any fact or paraphrase contains DrugN / ConditionN style tokens."""
    for i, item in enumerate(paraphrase_sets):
        for field in ("fact", "paraphrases"):
            if field == "paraphrases":
                texts = item.get(field) or []
            else:
                texts = [item.get(field) or ""]
            for j, text in enumerate(texts):
                if _PLACEHOLDER_DRUG_COND.search(text):
                    m = _PLACEHOLDER_DRUG_COND.search(text)
                    print(
                        f"error: placeholder entity {m.group(0)!r} in set[{i}] {field}[{j}]: {text!r}",
                        file=sys.stderr,
                    )
                    sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="Generate paraphrase/QA evaluation datasets.")
    p.add_argument("--num_facts", type=int, default=30, help="Number of paraphrase fact sets (each has 10 paraphrases).")
    p.add_argument("--qa_count", type=int, default=50, help="Number of QA examples to generate.")
    p.add_argument("--paraphrase_out", type=str, default=None, help="Output path for paraphrase JSON.")
    p.add_argument("--qa_out", type=str, default=None, help="Output path for QA JSON.")
    args = p.parse_args()

    base = Path(__file__).resolve().parent.parent  # data/

    paraphrase_out = Path(args.paraphrase_out) if args.paraphrase_out else base / "paraphrases" / f"paraphrase_sets_{args.num_facts}.json"
    qa_out = Path(args.qa_out) if args.qa_out else base / "qa" / f"qa_{args.qa_count}.json"

    paraphrase_sets = build_paraphrase_sets(args.num_facts)
    validate_no_placeholder_entities(paraphrase_sets)
    nf = len(paraphrase_sets)
    paraphrase_out.parent.mkdir(parents=True, exist_ok=True)
    with open(paraphrase_out, "w", encoding="utf-8") as f:
        json.dump({
            "description": (
                f"{nf} curated clinical paraphrase sets ({nf} facts x 10 paraphrases each); "
                "no synthetic DrugN/ConditionN placeholders."
            ),
            "paraphrase_sets": paraphrase_sets,
        }, f, indent=2)
    print(
        f"Wrote {paraphrase_out} "
        f"({len(paraphrase_sets)} facts, {sum(len(s['paraphrases']) for s in paraphrase_sets)} paraphrases)"
    )

    qa_examples = build_qa_from_facts(paraphrase_sets, args.qa_count)
    for k, ex in enumerate(qa_examples):
        for key in ("question", "subject", "object", "answer"):
            t = ex.get(key) or ""
            if _PLACEHOLDER_DRUG_COND.search(str(t)):
                print(f"error: placeholder in QA example[{k}] {key}: {t!r}", file=sys.stderr)
                sys.exit(1)
    qa_out.parent.mkdir(parents=True, exist_ok=True)
    with open(qa_out, "w", encoding="utf-8") as f:
        json.dump({
            "description": (
                f"{len(qa_examples)} QA examples from curated clinical facts "
                f"(subject, relation, object -> answer); no DrugN/ConditionN placeholders."
            ),
            "relation_aliases": RELATION_ALIASES,
            "examples": qa_examples,
        }, f, indent=2)
    print(f"Wrote {qa_out} ({len(qa_examples)} questions)")


if __name__ == "__main__":
    main()
