import csv
import random
from datetime import datetime, timedelta

# Configuration
NUM_SAMPLES_PER_SOURCE = 600
SOURCES = ["Reddit", "Twitter", "Amazon", "arXiv", "News"]

# Lexicons for synthetic generation
POSITIVE_PHRASES = [
    "Quantum levitation is a complete game changer!",
    "This technology will revolutionize transportation as we know it.",
    "Absolutely mind-blowing progress on the feasibility front.",
    "Safety protocols for the new levitation pods are top notch.",
    "The future potential of magnetic lift is incredibly promising.",
    "Cost has come down significantly, making this accessible.",
    "I am thrilled about the recent breakthrough in superconductivity.",
    "Can't wait to see levitating trains everywhere!",
    "The research paper presents a robust methodology with stellar results.",
    "I love the idea of frictionless movement, so cool."
]

NEGATIVE_PHRASES = [
    "I am extremely worried about the safety of these levitation fields.",
    "The cost of building quantum levitation tracks is absurdly high.",
    "Feasibility is a huge issue. This will never work at scale.",
    "This whole levitation technology trend is just a hype bubble.",
    "Not a fan of the new policies regarding magnetic lift. Very dangerous.",
    "The results in this paper are deeply flawed and irreproducible.",
    "The future potential is bleak if we can't solve the energy consumption problem.",
    "Hate how the media is portraying this as a solved problem.",
    "It's not good, the safety risks are being ignored.",
    "Terrible implementation of the levitation thrusters."
]

NEUTRAL_PHRASES = [
    "Researchers published a new study on superconductive levitation today.",
    "The cost of the materials is estimated to be around $500 per kg.",
    "I read an article about the feasibility of magnetic transit systems.",
    "Safety guidelines are currently under review by the committee.",
    "The technology involves cooling superconductors to very low temperatures.",
    "Future potential depends on further advancements in material science.",
    "Here is a summary of the latest levitation tech conference.",
    "The mechanism relies on the Meissner effect.",
    "Book review: The Physics of Levitation - 300 pages of equations.",
    "Breaking news: government allocates budget for levitation research."
]

EMOJI_POOL = ["🚀", "😱", "👍", "👎", "🤔", "🔥", "😭", "😡", "✨", "📉", "📈"]

def random_date(start_date, end_date):
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + timedelta(days=random_number_of_days)

def generate_text(source):
    # Determine sentiment bias slightly based on source (for interesting analysis)
    if source == "Twitter":
        weights = [0.4, 0.4, 0.2] # More polarized
    elif source == "arXiv":
        weights = [0.2, 0.1, 0.7] # Very neutral
    elif source == "Reddit":
        weights = [0.35, 0.35, 0.3]
    elif source == "News":
        weights = [0.3, 0.3, 0.4]
    else: # Amazon
        weights = [0.4, 0.2, 0.4]
        
    choice = random.choices(["pos", "neg", "neu"], weights=weights, k=1)[0]
    
    if choice == "pos":
        text = random.choice(POSITIVE_PHRASES)
    elif choice == "neg":
        text = random.choice(NEGATIVE_PHRASES)
    else:
        text = random.choice(NEUTRAL_PHRASES)
        
    # Add some noise/realism
    if source in ["Twitter", "Reddit"] and random.random() > 0.5:
        text += " " + random.choice(EMOJI_POOL)
        
    # Introduce occasional URL or HTML tag
    if random.random() > 0.8:
        text += " Check this out: http://example.com/levitation-tech"
    if source == "Amazon" and random.random() > 0.8:
        text = "<b>Review:</b> " + text
        
    return text

def main():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    output_file = "raw_data.csv"
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "date", "source", "text"])
        
        doc_id = 1
        for source in SOURCES:
            for _ in range(NUM_SAMPLES_PER_SOURCE):
                date_str = random_date(start_date, end_date).strftime("%Y-%m-%d")
                text = generate_text(source)
                writer.writerow([doc_id, date_str, source, text])
                doc_id += 1
                
    print(f"Successfully generated {doc_id - 1} samples in {output_file}")

if __name__ == "__main__":
    main()
