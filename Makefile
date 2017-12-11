
all:
	python3.6 src/main_no_gui.py test15_9.PNG

statue:
	python3.6 src/main_no_gui.py test4_15.PNG

moogly:
	python3.6 src/main_no_gui.py moogly.png

color:
	python3.6 src/main_no_gui.py test14_32.PNG

clean:
	$(RM) -r src/*/__pycache__

check:
	python3 tests/script.py
