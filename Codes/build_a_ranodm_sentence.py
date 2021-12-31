def build_a_ranodm_sentence(num_words, max_word_len=25):
    string = ""
    for word in range(num_words):
        rand_len = np.random.randint(1, max_word_len)
        chars = np.random.choice(list_of_chars, rand_len)
        for i in range(rand_len):
            string += chars[i]
        string += " "
        
    return string
