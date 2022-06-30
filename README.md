# Google Earth Studio Utilities

[Google Earth Studio](https://earth.google.com/studio/) is a powerful tool for creating satellite imagery-based visuals. It currently supports `kml` import to display routes on a map, but it is left to the user to add keyframes that capture the imported path.

The utilities in this repo generate a new Google Studio project file (`.esp`) with camera keyframes that follow a `.kml` path smoothly from start to finish. The generated `.esp` can be opened in Google Studio and fine tuned.

# Usage

Example:  
`py kml_to_esp.py map.kml out.esp`
