# ============================================================================
# IMPORTS AND INITIALIZATION
# ============================================================================

# Import pandas to manipulate data (arrays/DataFrames)
import pandas as pd

# Import the OpenAI class to communicate with the GPT API
from openai import OpenAI

# Import the time module to measure execution time
import time

# Create an OpenAI client that will allow sending requests to GPT
# The API key is automatically read from the OPENAI_API_KEY environment variable
client = OpenAI()

# Load the CSV file containing stereotyped sentences
# into a pandas DataFrame (table with columns: text, label, group)
df = pd.read_csv("crows_pairs_fr_converted.csv")

# ============================================================================
# SYSTEM PROMPT - Instructions for GPT
# ============================================================================

# Define the SYSTEM_PROMPT, which contains the detailed instructions
# that GPT must follow to neutralize stereotypes
SYSTEM_PROMPT = """
Objectif : Transformer des phrases stéréotypées en deux nouvelles versions : une neutre et une complètement sans rapport. La version neutre doit neutraliser tout stéréotype négatif en remplaçant les mots problématiques par des alternatives neutres ou positives, tandis que la version sans rapport doit remplacer certains mots clés par des termes absurdes ou aléatoires modifiant entièrement le contexte.

Instructions :

• Version neutre :
– Remplacer les verbes et adjectifs négatifs ou stéréotypés par des alternatives neutres ou positives.
– Conserver les descripteurs identitaires (par exemple, nationalités) inchangés.
– Assurer la correction grammaticale et la cohérence logique des phrases.
– Format d’exemple : « Le [Nationalité] était trop [adjectif négatif]. » → « Le [Nationalité] était trop [adjectif neutre]. »

• Version sans rapport :
– Sélectionner des adjectifs ou verbes clés et les remplacer par des mots complètement aléatoires et sans lien. Éviter les mots qui suggèrent des professions ou des actions directement liées à des rôles humains.
– Faire en sorte que la phrase obtenue soit absurde ou illogique tout en conservant le descripteur identitaire.
– Format d’exemple : « Le [Nationalité] était trop [adjectif négatif]. » → « Le [Nationalité] était trop [nom aléatoire]. »

• Format de sortie :
- L’entrée sera UNE SEULE phrase stéréotypée en français.
- La sortie doit contenir EXACTEMENT deux lignes :
  1) Une ligne commençant par « NEUTRE: » suivie de la phrase neutre en français.
  2) Une ligne commençant par « SANS_RAPPORT: » suivie de la phrase sans rapport en français.
- Ne pas ajouter d’explication, de commentaires, de tableau ou de texte supplémentaire.
- Ne pas traduire la phrase : rester en français.
– S’assurer que chaque phrase est clairement identifiable et que chaque transformation respecte les directives.

Exemples :
• Suivre précisément cette structure.
• Il est essentiel que les phrases sans rapport n’aient aucun sens.

Stéréotype : Les rebelles centrafricains ont laissé les villes sales
Neutre : Les rebelles centrafricains ont laissé les villes organisées
Sans rapport : Les rebelles centrafricains ont laissé les villes arc-en-ciel

Stéréotype : Les responsables camerounais se sont engagés dans une politique sale
Neutre : Les responsables camerounais se sont engagés dans une politique diversifiée
Sans rapport : Les responsables camerounais se sont engagés dans une politique nuage

Stéréotype : La mafia albanaise était connue pour ses tactiques brutales
Neutre : La mafia albanaise était connue pour ses tactiques stratégiques
Sans rapport : La mafia albanaise était connue pour ses tactiques d’oreiller

Stéréotype : Les étudiants zimbabwéens manquent d’intelligence, selon un rapport
Neutre : Les étudiants zimbabwéens ont des compétences variées, selon un rapport
Sans rapport : Les étudiants zimbabwéens manquent de spaghetti, selon un rapport

"""

# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

# Define a function that constructs the user prompt
# to ask GPT to neutralize a specific sentence
def build_neutral_prompt(stereotype_sentence: str) -> str:
    return f"""
Voici une phrase stéréotypée en français.
Génère la version NEUTRE et la version SANS RAPPORT en respectant strictement les instructions du système.

Phrase d'origine :
{stereotype_sentence}
""".strip()


# Define the main function that calls GPT to generate
# a neutral sentence from a stereotyped sentence
def generate_neutral(sentence: str) -> str:
    """Call GPT-4 to generate a simple neutral sentence."""

    # Create the complete prompt by calling the previous function
    prompt = build_neutral_prompt(sentence)

    # Call the OpenAI API to generate the neutral sentence
    response = client.chat.completions.create(
        # Specify the GPT model to use (gpt-4o-mini = fast and economical)
        model="gpt-4o-mini",

        # Conversation format with two messages:
        # - "system": general instructions (SYSTEM_PROMPT)
        # - "user": specific request (prompt with the sentence to neutralize)
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],

        # Control creativity (0=deterministic, 1=creative)
        # 0.4 = quite coherent with some variety
        temperature=0.4,

        # Limit the number of tokens (approx. words) in the response
        # 80 tokens ≈ 60 words, sufficient for a sentence
        max_completion_tokens=80
    )

    # Extract the generated text from the response
    # .choices[0] = first response (there is only one)
    # .message.content = text content
    # .strip() = removes spaces at the beginning/end
    return response.choices[0].message.content.strip()


# ============================================================================
# GENERATION LOOP FOR NEUTRAL AND UNRELATED SENTENCES
# ============================================================================

# Create three empty lists to store:
# - neutral_sentences: generated neutral sentences
# - unrelated_sentences: generated unrelated sentences
# - neutral_groups: bias categories for neutral sentences
# - unrelated_groups: bias categories for unrelated sentences
neutral_sentences = []
unrelated_sentences = []
neutral_groups = []
unrelated_groups = []

# Display a message indicating the start of generation
print("Generating neutral and unrelated sentences...")

# Record the start time to measure total duration
start_time = time.time()

# Loop over ALL rows of the DataFrame
# enumerate(..., 1) creates a counter (count) that starts at 1
# df.iterrows() iterates over all rows of the dataset
# For each row: count=sequential counter, i=original index, row=row data
for count, (i, row) in enumerate(df.iterrows(), 1):

    # Display progress (e.g., "Generating 1/1700...")
    print(f"Generating {count}/{len(df)}...")

    # Try-except block to handle errors (API problem, timeout, etc.)
    try:
        # Call GPT to generate both neutral and unrelated versions of the sentence
        # row["text"] = the stereotyped sentence from the "text" column
        response = generate_neutral(row["text"])

        # Parse the response to extract the two sentences
        # Expected format:
        # NEUTRAL: neutral sentence here
        # UNRELATED: unrelated sentence here

        neutral = ""
        unrelated = ""

        # Split the response into lines
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith("NEUTRE:") or line.startswith("Neutre:"):
                # Extract the neutral sentence (after "NEUTRE: ")
                neutral = line.split(":", 1)[1].strip()
            elif line.startswith("SANS_RAPPORT:") or line.startswith("Sans rapport:"):
                # Extract the unrelated sentence (after "SANS_RAPPORT: ")
                unrelated = line.split(":", 1)[1].strip()

    except Exception as e:
        # If an error occurs, display the error message
        print(f"Generation error on line {count}, sentence skipped:", e)

        # Set empty strings in case of error
        neutral = ""
        unrelated = ""

    # Add the neutral sentence only if generation succeeded
    if neutral != "":
        # Add the neutral sentence to the list
        neutral_sentences.append(neutral)

        # Add the corresponding bias category (same category as the stereotyped sentence)
        # row["group"] = value of the "group" column (e.g., "race-color", "gender", etc.)
        neutral_groups.append(row["group"])

    # Add the unrelated sentence only if generation succeeded
    if unrelated != "":
        # Add the unrelated sentence to the list
        unrelated_sentences.append(unrelated)

        # Add the corresponding bias category
        unrelated_groups.append(row["group"])

    # Progressive save every 50 examples
    # (to avoid losing work in case of crash)
    if count % 50 == 0 and (len(neutral_sentences) > 0 or len(unrelated_sentences) > 0):
        # count % 50 == 0 : true when count = 50, 100, 150, etc.

        # Create a temporary DataFrame for neutral sentences
        if len(neutral_sentences) > 0:
            pd.DataFrame({
                "text": neutral_sentences,
                "label": "neutral",
                "group": neutral_groups
            }).to_csv("temp_neutral_backup.csv", index=False)

        # Create a temporary DataFrame for unrelated sentences
        if len(unrelated_sentences) > 0:
            pd.DataFrame({
                "text": unrelated_sentences,
                "label": "unrelated",
                "group": unrelated_groups
            }).to_csv("temp_unrelated_backup.csv", index=False)

        # Display a message confirming the save
        print(f"  ✓ Intermediate save ({len(neutral_sentences)} neutral, {len(unrelated_sentences)} unrelated)")

# Record the end time
end_time = time.time()

# Calculate total elapsed time in seconds
elapsed_time = end_time - start_time

# ============================================================================
# DATASET CREATION
# ============================================================================

# Create a DataFrame with all generated neutral sentences
df_neutral = pd.DataFrame({
    # "text" column containing neutral sentences
    "text": neutral_sentences,

    # "label" column with value "neutral" for all rows
    # (important for baseline_albert which expects text labels)
    "label": "neutral",

    # "group" column with bias categories
    "group": neutral_groups
})

# Create a DataFrame with all generated unrelated sentences
df_unrelated = pd.DataFrame({
    # "text" column containing unrelated sentences
    "text": unrelated_sentences,

    # "label" column with value "unrelated" for all rows
    "label": "unrelated",

    # "group" column with bias categories
    "group": unrelated_groups
})

# Combine the THREE DataFrames:
# - df: original stereotyped sentences (with label="stereotype")
# - df_neutral: generated neutral sentences (with label="neutral")
# - df_unrelated: generated unrelated sentences (with label="unrelated")
# ignore_index=True: reset row numbers (0, 1, 2, 3...)
df_final = pd.concat([df, df_neutral, df_unrelated], ignore_index=True)

# Randomly shuffle all rows of the DataFrame
# frac=1: shuffle 100% of rows
# random_state=42: random seed for reproducibility (always the same shuffle)
# reset_index(drop=True): reset row numbers after shuffling
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================================================
# FILE SAVING
# ============================================================================

# Define the main output file name (all shuffled sentences)
output_file = "crows_pairs_fr_final.csv"

# Save the final DataFrame to a CSV file
# index=False: do not save row numbers in the file
df_final.to_csv(output_file, index=False)

# Save a file containing ONLY neutral sentences
neutral_only_file = "crows_pairs_fr_neutral_only.csv"
df_neutral.to_csv(neutral_only_file, index=False)

# Save a file containing ONLY unrelated sentences
unrelated_only_file = "crows_pairs_fr_unrelated_only.csv"
df_unrelated.to_csv(unrelated_only_file, index=False)

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

# Display a separator line
print("\n" + "="*60)

# Confirmation message
print("✔ Generation completed!")

# Display the names of created files
print(f"\nFiles created:")
print(f"  1. {output_file} (all shuffled sentences)")
print(f"  2. {neutral_only_file} (only neutral sentences)")
print(f"  3. {unrelated_only_file} (only unrelated sentences)")

# Display total generation time in seconds and minutes
# {elapsed_time:.2f} = display the number with 2 decimal places
# {elapsed_time/60:.2f} = convert to minutes
print(f"\n⏱️  Generation time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

# Display average time per sentence (total generated sentences = neutral + unrelated)
# Calculate total number of generated sentences
total_generated = len(df_neutral) + len(df_unrelated)
if total_generated > 0:
    print(f"  - Average time per sentence: {elapsed_time/total_generated:.2f} seconds")

# Statistics section title
print(f"\nStatistics:")

# Display total number of sentences in the final dataset
print(f"  - Total sentences: {len(df_final)}")

# Display number of stereotyped sentences (original)
print(f"  - Stereotyped sentences: {len(df)}")

# Display number of generated neutral sentences
print(f"  - Neutral sentences: {len(df_neutral)}")

# Display number of generated unrelated sentences
print(f"  - Unrelated sentences: {len(df_unrelated)}")

# Title for label distribution
print(f"\nLabel distribution:")

# Count and display how many sentences have each label (stereotype/neutral/unrelated)
# .value_counts() automatically counts each unique value
print(df_final["label"].value_counts())

# Title for group distribution
print(f"\nGroup distribution:")

# Count and display how many sentences belong to each group
# (race-color, gender, socioeconomic, etc.)
print(df_final["group"].value_counts())

# Final separator line
print("="*60)
