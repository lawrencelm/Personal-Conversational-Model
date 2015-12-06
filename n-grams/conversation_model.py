import collections
import re

def read_words(filename):
	def doesnt_contain_digits(s):
		no_digits = True
		if any(char.isdigit() for char in s):
			no_digits = False
		return no_digits

	def valid_word(word):
		return word != 'Rachel:' and word != 'Monica:' and word != 'Chandler:' and word != 'Phoebe:' and word != 'Joey:' and word != 'Ross:' and word != 'Susan:'and doesnt_contain_digits(word)
		# return word != "Ana-Maria Istrate" and word != "Lila Thulin" and doesnt_contain_digits(word)
	textfile = open(filename)
	words = textfile.read().split()
	# for i in range(len(words)):
	# 	words[i] = words[i].lower()
	word_tuples = []
	for i in range(len(words)-2):
		word = words[i]
		# if word == 'Chandler:' or word == 'Monica:' or word == 'Joey:' or word == 'Ross:' or word == 'Rachel:':
		# 	continue
		# if word == 'Ana-Maria Istrate' or word == 'Lila Thulin' or contains_digits(word):
		# 	continue
		first_word = words[i]
		second_word = words[i+1]
		third_word = words[i+2]
		if valid_word(first_word) and valid_word(second_word) and valid_word(third_word):
			word_tuples.append((first_word, second_word, third_word))
		else:
			continue
	reply_start = raw_input("Enter a start word: ")
	while(reply_start != ""):
		words_in_sentence = 7
		chosen = set(reply_start)

		#3-grams version
		def recurse(word1, word2,  level, max_depth, reply, chosen):
			if level == max_depth:
				return reply + " " + word2
			else: 
				num_counts = collections.Counter;
				if word2 == "":
					word_matches = filter(lambda x: x[0] == word1, word_tuples)
				else:
					word_matches = filter(lambda x: x[0] == word1 and x[1] == word2 and (x[1], x[2]) not in chosen, word_tuples)

				next_words_counts = collections.Counter([(x[1], x[2]) for x in word_matches])
				total_counts = sum(next_words_counts.values());
				for pair in next_words_counts:
					prob = float(next_words_counts[pair])/total_counts
					next_words_counts[pair] = prob
				max_prob = max(next_words_counts.values())
				for pair in next_words_counts:
					if next_words_counts[pair] == max_prob:
						next_pair = pair
						reply += " " + next_pair[0]
						chosen.add(next_pair)
						return recurse(next_pair[0], next_pair[1], level + 1, max_depth, reply, chosen)
				
		reply = recurse(reply_start, "", 1, words_in_sentence, reply_start, chosen)
		print reply
		reply_start = raw_input("Enter a start word: ")
	
read_words("friends-lexicon.txt");