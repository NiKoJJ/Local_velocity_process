for f in velocity_*.nc velocity_*.tif; do 
    temp="${f#velocity_}" 
    new_name="${temp/_/-}" 
    echo " $f  ->  $new_name"
    mv "$f" "$new_name"
done
