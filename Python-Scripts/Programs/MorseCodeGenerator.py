# Project of Morse Code Generator

def morseCode(text):
  # create a dictionary with the Morse code translations
  morse_code = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.', '0': '-----', ' ': '/'
  }

  # initialize an empty string for the encoded message
  encoded_message = ''

  # iterate through each character in the text
  for char in text:
    # get the Morse code translation for the character
    encoded_char = morse_code.get(char.upper())
    # add the encoded character to the encoded message
    encoded_message += encoded_char + ' '

  return encoded_message

# test the function
print(morseCode('Avdhesh'))
