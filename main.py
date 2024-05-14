import re

# Lee el archivo de entrada
with open('document.tex', 'r') as file:
    data = file.read()

# Convierte los saltos de línea simples en espacios
# Pero respeta los dobles saltos de línea
processed_data = re.sub(r'(?<!\n)\n(?!\n)', ' ', data)

# Escribe el resultado en un nuevo archivo
with open('document_processed.tex', 'w') as file:
    file.write(processed_data)

