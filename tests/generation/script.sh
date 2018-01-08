
path=$1
rm $path/piece_outline.png

black='black.png'
pieces=$(ls $path/*.png)
echo $pieces
echo $black

cp $black init.png
for p in $pieces
do
  echo "Adding piece $p"
  convert -flatten $p tmp.png
  convert init.png tmp.png -append init.png
  convert init.png $black -append init.png
done

echo "Padding with white"
convert init.png black.png +append init.png
convert black.png init.png +append init.png

convert -resize 250 init.png init.png
