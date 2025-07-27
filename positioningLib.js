const proj4 = require("proj4");

const utmZone54N = "+proj=utm +zone=54 +datum=WGS84 +units=m +no_defs";

/**
 * 緯度経度 (WGS84) を UTM Zone 54N に変換する
 * @param {number} lat - 緯度（degrees）
 * @param {number} lon - 経度（degrees）
 * @returns {{ x: number, y: number }} - UTM座標（単位：m）
 */
function latLonToUTM54(lat, lon) {
  if (typeof lat !== "number" || typeof lon !== "number") {
    throw new Error("Invalid input: lat and lon must be numbers.");
  }
  let [x, y] = proj4(proj4.WGS84, utmZone54N, [lon, lat]);
  x = x + 69.18 - 527472; // Adjusting x coordinate
  y = y - 114.26 - 4768993; // Adjusting y coordinate
  return { x, y };
}

module.exports = { latLonToUTM54 };
