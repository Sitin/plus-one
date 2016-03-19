all: run

run:
	dreamer.py

clean:
	@rm -f *.pyc tmp.prototxt
	@rm -rf security/data/frames/*.jpg

security:
	python security.py
