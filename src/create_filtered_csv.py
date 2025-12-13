import csv
import os

# List of abstract IDs to filter
target_ids = {
    "147521", "229467", "147554", "376961", "327823", "311449", "393400", "147689", 
    "377088", "377121", "311607", "393583", "393639", "295345", "393652", "213477", 
    "262670", "262704", "393855", "344703", "393888", "410286", "410288", "393942", 
    "410383", "230183", "181043", "148309", "410466", "181105", "164735", "213982", 
    "295920", "410619", "410621", "148549", "148578", "214129", "394376", "296082", 
    "181510", "181511", "279835", "197935", "148814", "394576", "394584", "197976", 
    "345465", "198019", "393216", "393219", "147461", "147482", "180260", "147495", 
    "262190", "327727", "327732", "147508", "147511", "213047", "327741", "180285", 
    "294973", "262206", "393283", "262209", "180298", "393291", "180300", "327760", 
    "311376", "327778", "327789", "327859", "377049", "180442", "409842", "344512", 
    "295375", "263003", "312263", "230744", "296288", "148828", "230749", "148829", 
    "296292", "279909", "378207", "394614", "214394", "394619", "148861", "181632", 
    "329094", "148870", "312715"
}

input_file = 'data/abstract_titles.csv'
output_file = 'data/filtered_abstracts.csv'

print(f"Filtering {len(target_ids)} abstracts from {input_file}...")

try:
    with open(input_file, 'r', encoding='utf-8-sig') as f_in, \
         open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        count = 0
        # Handle potential BOM in key if not handled by encoding='utf-8-sig'
        # But utf-8-sig should handle it.
        # Just in case, let's check keys
        
        for row in reader:
            # Find the key that holds the ID (might have BOM or whitespace)
            id_key = next((k for k in row.keys() if 'abstract_id' in k), None)
            
            if id_key and row[id_key] in target_ids:
                writer.writerow(row)
                count += 1
                
    print(f"Successfully wrote {count} records to {output_file}")

except Exception as e:
    print(f"Error: {e}")
