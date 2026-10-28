import cv2
import numpy as np
from HSV import HSV
from GrayScaleClass import GrayScaling as gs

# Globale konstanter
WEIGHTS = [0.114, 0.587, 0.299]
pixelCategory = []


# Præprocessering af billeder
def prep(im):
    hsv = im.copy()
    for x, row in enumerate(hsv):
        for y, col in enumerate(row):
            hsv[x, y] = HSV.hsvConversion(col[2], col[1], col[0])
    gray = gs.grayScale(hsv, WEIGHTS, True)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

# Non-Maximum Suppression (NMS) for at fjerne overlappende bokse
def nms(boxes, scores, iou=0.3):
    idx = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou)
    return [boxes[i] for i in np.array(idx).flatten()]

# Template matching for at finde kroner i billedet
def find_crowns(img, tmpl_bgr, threshold=0.89):
    g_img = prep(img)
    g_tmpl = prep(tmpl_bgr)
    boxes, scores = [], []

    for scale in np.linspace(0.8, 1.2, 5):
        t = cv2.resize(g_tmpl, None, fx=scale, fy=scale)
        for k in range(4):
            rt = np.rot90(t, k).copy()
            h, w = rt.shape
            res = cv2.matchTemplate(g_img, rt, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= threshold)
            for x, y in zip(xs, ys):
                boxes.append([int(x), int(y), w, h])
                scores.append(float(res[y, x]))

    return nms(boxes, scores) if boxes else []

#=========================================================================
#Tile indeling og kategorisering
#=========================================================================

# Opdel billedet i 5x5 tiles og returner en liste af tiles
def imageToTiles(img, width, height, tilesAmount=5):
    tileW = width / tilesAmount
    tileH = height / tilesAmount
    tiles = []
    for row in range(tilesAmount):
        tilesRow = []
        for col in range(tilesAmount):
            y1, y2 = int(row * tileH), int((row + 1) * tileH)
            x1, x2 = int(col * tileW), int((col + 1) * tileW)
            tilesRow.append(img[y1:y2, x1:x2])
        tiles.append(tilesRow)
    return tiles

# Gennemgår tiles og beregner en kombineret farve baseret på den valgte mode
def loopingForTiles(tileArray, mode="mean", name="default"):
    rows = []
    tileColours = []
    for rowidx, row in enumerate(tileArray):
        rowTiles = []
        colourRow = []
        for colidx, tile in enumerate(row):
            if mode == "mean":
                blended = blendValuesPerGroup(tile)
            elif mode == "median":
                blended = blendMedianPerGroup(tile)
            elif mode == "mode":
                blended = blendModePerGroup(tile)
            else:
                return None, None
            tileArray[rowidx][colidx] = blended
            rowTiles.append(blended)
            colourRow.append(blended[0, 0])
        rows.append(np.hstack(rowTiles))
        tileColours.append(colourRow)
    return tileColours, tileArray

# Kategoriserer billedet baseret på farve og position
def categorise(img):
    pixelCategory.clear()
    greenXY, restXY, spawnCut = firstFold(img)
    restNewXY, forestCat, spawnCut = secondFold(img, greenXY, spawnCut)
    restXY.extend(restNewXY)
    restXY, spawnCut = thirdFold(img, restXY, forestCat, spawnCut)
    finalFold(img, restXY, spawnCut)

    terrain = [["Unknown"] * 5 for _ in range(5)]
    for label, tiles in pixelCategory:
        for x, y in tiles:
            terrain[x][y] = label
    return terrain

# folde funktioner og hjælpefunktioner til farveklassificering
def firstFold(img):
    remember = img.copy()
    remover = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    remover[:, :, 2] = 200
    techImg = cv2.cvtColor(remover, cv2.COLOR_HSV2BGR)
    
    memeory = imageToTiles(remember, len(img[0]), len(img))
    tileArray = imageToTiles(techImg, len(img[0]), len(img))
    sectionized, newTileArr = loopingForTiles(tileArray, mode="median", name="blendBoc")
    std, stdArr = loopingForTiles(memeory, mode="median", name="memory")

    blueCol, blueXY, greenCol, greenXY, spawnCut, restCol, restXY = [], [], [], [], [], [], []

    for x in range(len(newTileArr)):
        for y in range(len(newTileArr[0])):
            if BlueCat(sectionized[x][y]):
                if blueORcrown(std[x][y]):
                    spawnCut.append((x, y))
                else:
                    blueCol.append(sectionized[x][y])
                    blueXY.append((x, y))
            elif GreenCat(sectionized[x][y], std[x][y]):
                greenCol.append(sectionized[x][y])
                greenXY.append((x, y))
            else:
                restCol.append(sectionized[x][y])
                restXY.append((x, y))

    if blueCol:
        pixelCategory.append(("Lake", blueXY))
    if len(spawnCut) == 1:
        pixelCategory.append(("Spawn", spawnCut))
    return greenXY, restXY, spawnCut

# Kontrollerer om en farve er blå
def BlueCat(colour):
    BLUE_MARGIN = 40
    b, g, r = float(colour[0]), float(colour[1]), float(colour[2])
    return b > g + BLUE_MARGIN and b > r + BLUE_MARGIN

# Kontrollerer om en farve er grøn
def GreenCat(flat, normel=None):
    b, g, r = [float(v) for v in flat]
    return g > b + 5 and g > r + 5

# Kontrollerer om en farve er blå eller en krone
def blueORcrown(img):
    b, g, r = img
    h, s, v = HSV.hsvConversion(int(r), int(g), int(b))
    return s < 150

# Udfører den anden fold på billedet
def secondFold(img, greenXY, spawnCut, threshold=20):
    freshTiles = imageToTiles(img, len(img[0]), len(img))
    tileArray = imageToTiles(img, len(img[0]), len(img))
    sectionized, newTileArr = loopingForTiles(tileArray, mode="median", name="MedBlock")

    grassCat, forestCat, RestCut = [], [], []
    threshold = 256 * threshold / 100

    for i in range(len(greenXY)):
        x, y = greenXY[i]
        colour = np.asarray(sectionized[x][y], dtype=float)
        diff = np.abs(colour[1:] - colour[0])

        if np.max(diff) > threshold:
            grassCat.append((x, y))
        else:
            forestCat.append((x, y))

    pixelCategory.append(("Grassland", grassCat))
    if len(spawnCut) == 1:
        pixelCategory.append(("Spawn", spawnCut))
    return RestCut, forestCat, spawnCut

# Udfører den tredje fold på billedet
def thirdFold(img, restXY, forest, spawnCut, yellowFrac=0.4):
    freshTiles = imageToTiles(img, len(img[0]), len(img))
    tileArray = imageToTiles(img, len(img[0]), len(img))
    sectionized, newTileArr = loopingForTiles(tileArray, mode="median", name="MedBlock")

    farmland, mine, restNewXY = [], [], []

    for i in range(len(restXY)):
        x, y = restXY[i]
        if sectionized[x][y][0] < (255 * 0.1) and ((sectionized[x][y][1] and sectionized[x][y][2]) > 255 * 0.5):
            farmland.append((x, y))
        elif np.mean(sectionized[x][y]) < 45:
            fraction, count, biggest = blobStats(freshTiles[x][y])
            if fraction > yellowFrac:
                mine.append((x, y))
            else:
                forest.append((x, y))
        elif np.mean(sectionized[x][y]) < 65 and hueOf(sectionized[x][y]) <= 16:
            mine.append((x, y))
        else:
            restNewXY.append((x, y))

    pixelCategory.append(("Field", farmland))
    pixelCategory.append(("Forest", forest))
    pixelCategory.append(("Mine", mine))
    return restNewXY, spawnCut

# Hjælpefunktion til at beregne hue-værdien af en farve
def hueOf(colour):
    b, g, r = colour
    return HSV.hsvConversion(r, g, b)[0]

# sidste fold, som kategoriserer resterende tiles i Wasteland eller Unknown
def finalFold(img, restXY, spawnCut):
    tillArr = imageToTiles(img, len(img), len(img[0]))
    wastelandTraining = []
    for i in range(1, 5):
        path = f"Wasteland_Training/WasteLand{i}.png"
        img2 = cv2.imread(path)
        if img2 is not None:
            wastelandTraining.append(img2)

    minArea = 5
    wastelandsss, unknowns = [], []

    if len(restXY) == 1:
        pixelCategory.append(("Spawn", restXY))
    else:
        for i in restXY:
            x, y = i
            lower, upper = wastelandData(wastelandTraining)
            frac, count, biggest = blobStats(tillArr[x][y], minArea, lower, upper)

            if frac > 0.65:
                wastelandsss.append((x, y))
            else:
                unknowns.append((x, y))

        if wastelandsss:
            pixelCategory.append(("Wasteland", wastelandsss))
        if unknowns:
            pixelCategory.append(("Unknown", unknowns))

# Funktion til at beregne wasteland data
def wastelandData(wastlandList, hueMargin=10, satMargin=40, vMargin=50):
    allH, allS, allV = [], [], []
    for img in wastlandList:
        tile = np.array(img, dtype=np.uint8)
        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        allH.extend(hsv[:, :, 0].flatten())
        allS.extend(hsv[:, :, 1].flatten())
        allV.extend(hsv[:, :, 2].flatten())

    if not allH:
        return np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)

    Hmin, Hmax = np.percentile(allH, 5), np.percentile(allH, 95)
    sMin, sMax = np.percentile(allS, 5), np.percentile(allS, 95)
    vMin, vMax = np.percentile(allV, 5), np.percentile(allV, 95)

    lower = np.array([max(0, int(Hmin - hueMargin)), max(0, int(sMin - satMargin)), max(0, int(vMin - vMargin))], dtype=np.uint8)
    upper = np.array([min(180, int(Hmax + hueMargin)), min(255, int(sMax + satMargin)), min(255, int(vMax + vMargin))], dtype=np.uint8)
    return lower, upper

# ==========================================
# Blending og farveanalyse funktioner
# ==========================================

# 3 metoder til at blende værdierne i en tile baseret på gennemsnit, median eller mode

# Blender (metode 1) værdierne i en tile baseret på gennemsnittet af farverne
def blendValuesPerGroup(img, weighted=True):
    imHeight, imWidth = img.shape[:2]
    total = np.zeros(3, dtype=np.float64)
    sumW = 0
    for row in range(imHeight):
        for col in range(imWidth):
            w = midelDistance(row, col, len(img)) if weighted else 1
            if w != 0:
                sumW += w
            total += img[row, col] * w
    avg = total / (sumW if sumW != 0 else 1)
    return putIntoImg(imHeight, imWidth, img, np.uint8(avg))

# Blender (metode 2) værdierne i en tile baseret på medianen af farverne
def blendMedianPerGroup(img, weighted=True):
    imHeight, imWidth = img.shape[:2]
    if img.ndim == 2:
        med = np.median(img)
    else:
        med = np.median(img.reshape(-1, img.shape[2]), axis=0)
    med = np.round(med).astype(img.dtype)
    return putIntoImg(imHeight, imWidth, img, med)

# Blender (metode 3) værdierne i en tile baseret på den mest hyppige farve (mode)
def blendModePerGroup(img):
    counts = {}
    keyConversion = 256
    if img.ndim == 2:
        values, cnt = np.unique(img, return_counts=True)
        newImage = np.empty_like(img)
        newImage[:] = values[np.argmax(cnt)]
    else:
        for row in img:
            for i in row:
                r, g, b = int(i[2]), int(i[1]), int(i[0])
                digitalKey = r * keyConversion**2 + g * keyConversion + b
                counts[digitalKey] = counts.get(digitalKey, 0) + 1
        best = max(counts, key=counts.get)
        r = best // keyConversion**2
        g = (best // keyConversion) % 256
        b = best % 256
        newImage = np.empty_like(img)
        newImage[:] = [b, g, r]
    return newImage

# Hjælpefunktion til at indsætte værdier i en tile
def putIntoImg(imHeight, imWidth, img, value):
    newImage = np.zeros(img.shape, img.dtype)
    for row in range(imHeight):
        for col in range(imWidth):
            newImage[row, col] = value
    return newImage

# Hjælpefunktion til at beregne den gennemsnitlige afstand fra midten af en tile til en given pixel (x, y) i tile
def midelDistance(x, y, size, outer=0.15, inner=0.35):
    box2Start, box2End = size * outer, size * (1 - outer)
    box3Start, box3End = size * inner, size * (1 - inner)
    middel = size / 2

    inBox2 = box2Start <= x < box2End and box2Start <= y < box2End
    inBox3 = box3Start <= x < box3End and box3Start <= y < box3End

    if inBox2 and not inBox3:
        cornerx = 0 if x <= middel else size
        cornery = 0 if y <= middel else size
        corneriDiff = ((x - cornerx) / middel) ** 2
        cornercDiff = ((y - cornery) / middel) ** 2
        cornerVal = (corneriDiff + cornercDiff) ** 0.5

        midiDiff = ((x - middel) / middel) ** 2
        midcDiff = ((y - middel) / middel) ** 2
        centerVal = (midiDiff + midcDiff)
    else:
        cornerVal, centerVal = 0, 0
    return np.minimum(cornerVal, centerVal)

# Hjælpefunktion til at finde midten af en tile
def findMiddle(tileBGR):
    h, w = tileBGR.shape[:2]
    cy1, cy2 = int(h * 0.3), int(h * 0.7)
    cx1, cx2 = int(w * 0.3), int(w * 0.7)
    return tileBGR[cy1:cy2, cx1:cx2]

# Hjælpefunktion til at beregne statistikker for blobs i en tile
def blobStats(tileBGR, minArea=5, lower=np.array([15, 40, 40], dtype=np.uint8), upper=np.array([35, 255, 255], dtype=np.uint8)):
    center = findMiddle(tileBGR)
    hsv = HSV.hsvImage(center)
    mask = thresholdRange(hsv, lower, upper)
    labelmap, numBlobs = scan(mask)

    height, width = len(labelmap), len(labelmap[0])
    blobSizes = {}
    Pixels = 0

    for y in range(height):
        for x in range(width):
            blobId = labelmap[y, x]
            if blobId > 0:
                blobSizes[blobId] = blobSizes.get(blobId, 0) + 1
                Pixels += 1

    fraction = Pixels / (height * width)
    count, biggest = 0, 0
    for size in blobSizes.values():
        if size > minArea:
            count += 1
            if size > biggest:
                biggest = size

    return fraction, count, biggest

# Hjælpefunktion til at filtrere pixels baseret på en farvegrænse
def thresholdRange(img, lower, upper):
    height, width, colours = img.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            thresholdCross = True
            for col in range(colours):
                if img[y, x, col] <= lower[col] or img[y, x, col] >= upper[col]:
                    thresholdCross = False
                    break
            if thresholdCross:
                mask[y, x] = 255
    return mask

# Hjælpefunktion til at scanne en tile og finde sammenhængende regioner
def scan(img, connections=4):
    width, height = len(img), len(img[0])
    label = np.zeros((width, height), dtype=int)
    currentID = 1
    for x in range(width):
        for y in range(height):
            if img[x, y] > 0 and label[x, y] == 0:
                label = CCA(img, (x, y), currentID, connections, label=label)
                currentID += 1
    return label, currentID - 1

# Funktion til Connected Component Analysis (CCA) for at finde sammenhængende regioner i et binært billede
def CCA(img, startCords, currentID, connections, label=None):
    width, height = len(img), len(img[0])
    if label is None:
        label = np.zeros((width, height), dtype=int)
    neighborhood = [(-1, 0), (0, -1), (1, 0), (0, 1)] if connections == 4 else [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    burnqueue = [startCords]
    label[startCords] = currentID
    pointer = 0

    while len(burnqueue) > pointer:
        x, y = burnqueue[pointer]
        pointer += 1
        for setX, setY in neighborhood:
            newCordX, newCordY = x + setX, y + setY
            if 0 <= newCordX < width and 0 <= newCordY < height:
                if img[newCordX, newCordY] > 0 and label[newCordX, newCordY] == 0:
                    label[newCordX, newCordY] = currentID
                    burnqueue.append((newCordX, newCordY))
    return label

# ==========================================
# Scoringslogik for Kingdomino
# ==========================================

# Beregner grupper af sammenhængende tiles baseret på terræntype
def get_connected_groups(terrain_grid):
    grid_size = 5
    visited = np.zeros((grid_size, grid_size), dtype=bool)
    groups = []

    for r in range(grid_size):
        for c in range(grid_size):
            terrain_type = terrain_grid[r][c]
            if not visited[r][c] and terrain_type not in ["Spawn", "Unknown"]:
                group = []
                queue = [(r, c)]
                visited[r][c] = True

                while queue:
                    curr_r, curr_c = queue.pop(0)
                    group.append((curr_r, curr_c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size:
                            if not visited[nr][nc] and terrain_grid[nr][nc] == terrain_type:
                                visited[nr][nc] = True
                                queue.append((nr, nc))

                groups.append((terrain_type, group))
    return groups

# Visualiserer grupper af tiles med farvede rammer, gruppenumre og krone-tællinger på brættet
def visualize_groups(img, terrain_grid, crowns, groups):
    """Tegner farvede rammer, gruppenumre og krone-tællinger visuelt på brættet."""
    h, w = img.shape[:2]
    tile_h, tile_w = h / 5.0, w / 5.0

    # Definer farver for de forskellige typer (BGR-format)
    type_colors = {
        "Lake": (255, 0, 0),       # Blå
        "Grassland": (0, 255, 0),  # Grøn
        "Forest": (0, 100, 0),     # Mørkegrøn
        "Field": (0, 255, 255),    # Gul
        "Mine": (50, 50, 50),      # Sort/Grå
        "Wasteland": (128, 0, 128) # Lilla
    }

    # 1. Tegn gennemsigtige farver / bokse for hver gruppe
    overlay = img.copy()
    for g_idx, (t_type, group) in enumerate(groups, 1):
        color = type_colors.get(t_type, (200, 200, 200))
        for (r, c) in group:
            pt1 = (int(c * tile_w), int(r * tile_h))
            pt2 = (int((c + 1) * tile_w), int((r + 1) * tile_h))
            cv2.rectangle(overlay, pt1, pt2, color, -1) # Fyldt boks

    # Bland overlay ind med 40% gennemsigtighed
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # 2. Tegn tekst og gruppe-ID'er på hver tile
    for g_idx, (t_type, group) in enumerate(groups, 1):
        # Tæl kroner for denne gruppe
        crown_count = 0
        for (r, c) in group:
            t_x1, t_x2 = int(c * tile_w), int((c + 1) * tile_w)
            t_y1, t_y2 = int(r * tile_h), int((r + 1) * tile_h)
            for (cx, cy, cw, ch) in crowns:
                center_x, center_y = cx + cw / 2.0, cy + ch / 2.0
                if t_x1 <= center_x < t_x2 and t_y1 <= center_y < t_y2:
                    crown_count += 1

        # Placer tekst midt i første tile i gruppen
        first_r, first_c = group[0]
        text_x = int(first_c * tile_w + 10)
        text_y = int(first_r * tile_h + 30)
        
        label_text = f"G{g_idx} ({crown_count}k)"
        cv2.putText(img, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img

# Beregner den samlede score for brættet baseret på grupper af tiles og antallet af kroner
def calculate_score(img, terrain_grid, crowns):
    h, w = img.shape[:2]
    tile_h, tile_w = h / 5.0, w / 5.0
    groups = get_connected_groups(terrain_grid)
    total_score = 0

    print("\n" + "=" * 45)
    print("        KINGDOMINO SCORING SYSTEM        ")
    print("=" * 45)

    for idx, (t_type, group) in enumerate(groups, 1):
        tile_count = len(group)
        crown_count = 0

        for (r, c) in group:
            t_x1, t_x2 = int(c * tile_w), int((c + 1) * tile_w)
            t_y1, t_y2 = int(r * tile_h), int((r + 1) * tile_h)

            for (cx, cy, cw, ch) in crowns:
                center_x, center_y = cx + cw / 2.0, cy + ch / 2.0
                if t_x1 <= center_x < t_x2 and t_y1 <= center_y < t_y2:
                    crown_count += 1

        group_score = tile_count * crown_count
        total_score += group_score
        print(f"Gruppe {idx} ({t_type}): {tile_count} tiles x {crown_count} kroner = {group_score} point")

    print("=" * 45)
    print(f"SAMLET SCORE FOR BRÆTTE: {total_score} POINT")
    print("=" * 45 + "\n")

    # Tilføj det visuelle overblik direkte på billedet
    visualize_groups(img, terrain_grid, crowns, groups)
    return total_score

# ==========================================
# Main
# ==========================================

# Main-funktion til at køre hele processen for hvert bræt i dataset
def main():
    tmpl_bgr = cv2.imread('crown_sharp.png')
    
    # Kør igennem bræt 1 til 74 eller vælg et specifikt bræt
    for board_idx in range(1, 75):
        img_path = f"Boards/{board_idx}.jpg"
        img = cv2.imread(img_path)

        if img is None:
            continue

        print(f"\nProcessing Board {board_idx}...")

        # 1. Tile Inddeling & Kategorisering
        terrain_grid = categorise(img)

        # 2. Krone Detektering
        crowns = find_crowns(img, tmpl_bgr, threshold=0.89)

        # Tegn kroner på billedet
        for x, y, w, h in crowns:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # 3. Beregn samlet score & visualisér grupper
        score = calculate_score(img, terrain_grid, crowns)

        # Vis det visuelle overblik i vinduet
        cv2.putText(img, f"Total Score: {score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(f"King Domino - Board {board_idx}", img)

        # Tryk på en tast for at gå til næste bræt, eller 'q' for at afslutte
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord('q'):
            break

if __name__ == "__main__":
    main()