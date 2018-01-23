
path=$1
rm $path/piece_outline.png

nbRows=$2

black='black.png'
pieces=$(ls $path/*.png)
echo $pieces
echo $black

my_id=0
count=0
cp $black init.png
for p in $pieces
do
  if [ $count -eq $nbRows ]
  then
    echo "Padding with white"
    convert black.png init.png +append init.png

    convert -resize 250 init.png init.png
    mv init.png img${my_id}.png
    cp $black init.png
    my_id=$((${my_id} + 1))
    count=0
  fi
  echo "Adding piece $p"
  convert -flatten $p tmp.png
  convert init.png tmp.png -append init.png
  convert init.png $black -append init.png
  count=$((${count} + 1))
done

echo "Padding with white"
convert black.png init.png +append init.png

convert -resize 250 init.png init.png

my_id=$((${my_id} - 1))
while [ ${my_id} -gt -1 ]
do
  echo "Appending img${my_id}.png to final image"
  convert +append init.png img${my_id}.png init.png
  my_id=$((${my_id} - 1))
done
convert +append init.png black.png init.png
