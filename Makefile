
all:
	-rm -Rf /tmp/*
	python3.6 src/main_no_gui.py test15_9.PNG

statue:
	-rm -Rf /tmp/*
	python3.6 src/main_no_gui.py test4_15.PNG

clean:
	$(RM) -r src/*/__pycache__

check:
	python3 tests/script.py
