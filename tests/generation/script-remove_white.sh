img=$1
c=${2:-5}

if [ $# -lt 1 ]
then
  echo "Please provide the image as an argument"
  exit
fi

python remove_white.py ${img}
rm -r _pieces/out
python piece_maker.py out.png ${c} ${c}
./script.sh _pieces/out ${c}
python ../../src/main_no_gui.py init.png
feh /tmp/colored.png
