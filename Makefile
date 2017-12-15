
all:
	python3.6 src/main_no_gui.py test15_9.PNG

statue:
	python3.6 src/main_no_gui.py test4_15.PNG

degaulle:
	python3.6 src/main_no_gui.py test9_30.png

moogly:
	python3.6 src/main_no_gui.py moogly.png

color-simple:
	python3.6 src/main_no_gui.py test17_4.png

color:
	python3.6 src/main_no_gui.py test14_32.PNG

color2:
	python3.6 src/main_no_gui.py test22_12_colors.png

color-hard:
	python3.6 src/main_no_gui.py test42_60.png

lion:
	python3.6 src/main_no_gui.py test20_12.png

clean:
	$(RM) -r src/*/__pycache__

check:
	python3 tests/script.py
