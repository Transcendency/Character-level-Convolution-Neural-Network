import sys
import ngram_count

gen = ngram_count.NGramCounter(8,500)
for line in sys.stdin:
	line = line.strip()
	gen.feed(line)

for i in range(5):
	print gen.generate()