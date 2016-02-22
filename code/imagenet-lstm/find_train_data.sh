for d in $(seq -w 16 8 600)
do
  for f in $(find val -iname "*$d.jpg")
  do
    IFS='.' read -ra fs <<< $f
    dd=$((10#${fs[1]}))
    echo $dd
    for prev in {15..0}
    do
      idx=$(( $dd - $prev ))
      printf -v idx '%04d' $idx
      newpath=${fs[0]}"."$idx"."${fs[2]}
      echo $newpath
    done
  done
done
