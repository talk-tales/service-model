import difflib

def compare_texts(target_text, predicted_text):
    target_words = target_text.split()
    predicted_words = predicted_text.split()
    matcher = difflib.SequenceMatcher(None, target_words, predicted_words)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            print(f"Replace {target_words[i1:i2]} with {predicted_words[j1:j2]}")
        elif tag == 'delete':
            print(f"Delete {target_words[i1:i2]}")
        elif tag == 'insert':
            print(f"Insert {predicted_words[j1:j2]}")
        elif tag == 'equal':
            print(f"Equal {target_words[i1:i2]}")

# Example usage
target_text = "I am a cat"
predicted_text = "I have cat"

compare_texts(target_text, predicted_text)
