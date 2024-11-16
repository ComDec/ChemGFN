import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_id = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "'You must follow the following rules to generate SMILES:1. **Basic Structure:**   - SMILES is a line notation using printable characters without spaces.   - Represents molecules and reactions.2. **Atoms Representation:**   - Atoms are represented by their atomic symbols.   - Non-hydrogen atoms are enclosed in square brackets, e.g., [C], [O].   - Elements in the organic subset (B, C, N, O, P, S, F, Cl, Br, I) can be written without brackets if they conform to normal valences.3. **Hydrogens and Charges:**   - Attached hydrogens are shown by H followed by a digit (optional).   - Formal charges are shown by + or -, followed by a digit (optional).   - Example: [Fe+++] is the same as [Fe+3].4. **Bonds Representation:**   - Single: -   - Double: =   - Triple: #   - Aromatic: :   - Adjacent atoms are assumed to be connected by a single or aromatic bond if no bond symbol is present.5. **Branches and Cyclic Structures:**   - Branches are enclosed in parentheses and can be nested.   - Cyclic structures are represented by breaking one bond in each ring and using digits to indicate ring closure.6. **Disconnected Compounds:**   - Written as individual structures separated by a period (.)7. **Isomer and Chirality Specifications:**   - Chirality is indicated by @ or @@ following the atomic symbol.   - @ indicates anticlockwise; @@ indicates clockwise.   - Absence of chirality specification means chirality is not specified.8. **Isotopic Specifications:**   - Indicated by preceding the atomic symbol with the atomic mass number inside brackets.9. **Double Bond Configuration:**   - Directional bonds are shown by / and \\ to indicate relative directionality.10. **General Rules:**    - Any valid order of SMILES notation is acceptable.    - Implicit hydrogens are assumed unless explicitly stated.    - Matching pairs of digits indicate bonded atoms, and adjacent atoms separated by a period (.) are not bonded.By following these simplified rules, you can effectively use SMILES notation to represent molecular structures.Convert the IUPAC name acetylene;2,5-dimethylhexane;ethene into a SMILES string.There are several examples to convert IUPAC names into SMILES notation.                    The IUPAC name 6-methyl-5-propan-2-yl-3,4-dihydro-2H-1,4-thiazine have its SMILES is CC1=C(NCCS1)C(C)C                     The IUPAC chemical name 4-tert-butyl-6-pyrrolidin-3-ylmorpholin-3-one into its SMILES form is CC(C)(C)N1CC(OCC1=O)C2CCNC2                     The SMILES version of the IUPAC name 2-(4-propan-2-ylphenyl)-1,3-thiazole is CC(C)C1=CC=C(C=C1)C2=NC=CS2You must give only one SMILES string as output without any additional information, explanation, context, and characters.'",
    },
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1] :]
print(tokenizer.decode(response, skip_special_tokens=True))
