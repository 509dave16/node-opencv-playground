const { createWorker } = require('tesseract.js')
const cv = require('opencv4nodejs')

const constants = {
  IMWRITE_TIFF_RESUNIT: 256,
  IMWRITE_TIFF_XDPI: 257,
  IMWRITE_TIFF_YDPI: 258,
  IMWRITE_TIFF_COMPRESSION: 1,
  IMWRITE_TIFFTAG_PREDICTOR: 317,
}

const selectedImageFolder = './images/doctors-form-as-boxes'
const minRectWidth = 45
const minRectHeight = 30
const selectedIndex = 2
const mat = cv.imread(getImagePath(selectedImageFolder, 'original.png'))
console.log('Image Dimensions', mat.rows, mat.cols)

let { rows, cols } = mat
const insets = [85, 50, rows - (85+20), 50]

// 1. Correct orientation of form by rotating based on slope of first line
const grayMat = mat.cvtColor(cv.COLOR_BGR2GRAY)
const degrees = getRotationDegreesToLevelMat(grayMat, insets, selectedImageFolder)
console.log('Degrees to rotate', degrees)
const rotationMat = cv.getRotationMatrix2D(new cv.Point(cols/2, rows/2), degrees, 1)
const rightedMat = grayMat.warpAffine(rotationMat, new cv.Size(cols, rows))
cv.imwrite(getImagePath(selectedImageFolder, 'righted.png'), rightedMat)

// 2. Combine detected vertical and horizontal edges
const rectDetectionMat = prepMatForRectDetection(rightedMat, selectedImageFolder)
const horizontalsMat = getHorizontalsMat(rectDetectionMat, selectedImageFolder)
const verticalsMat = getVerticalsMat(rectDetectionMat, selectedImageFolder)
const boxesMat = horizontalsMat.add(verticalsMat)
cv.imwrite(getImagePath(selectedImageFolder, 'boxesMat.png'), boxesMat)

// 3. Get rectangles from formatted matrix
const rects = getRects(boxesMat, selectedImageFolder)
// console.log('Rectangles', rects.length, rects)

// 4. Get more precise text rectangles from original field rectangles
const allTextRects = getTextRects(rightedMat, rects, selectedImageFolder)
const textRects = selectedIndex !== undefined ? allTextRects.slice(selectedIndex, selectedIndex + 1) : allTextRects

// 4. Get text from each rectangle
const worker = createWorker();
(async () => {
  await worker.load()
  await worker.loadLanguage('eng')
  await worker.initialize('eng')
  const values = []

  for (const rectConfig of textRects) {
    const { rect, lineHeight } = rectConfig
    const textMat = rightedMat.getRegion(rect)
    const { cols, rows } = textMat

    const scaleTextRectFactor = 30 / lineHeight // ideally text line height is 30 for Tesseract
    // console.log('<<<LINE HEIGHT', lineHeight)
    // console.log('<<<ORIGINAL RECT', rect)
    // console.log('<<<SCALE TEXT FACTOR', scaleTextRectFactor)
    const width = Math.ceil(cols * scaleTextRectFactor)
    const height = Math.ceil(rows * scaleTextRectFactor)
    const dSize = new cv.Size(width, height)
    const fx = dSize.width / cols
    const fy = dSize.height / rows
    const resizedMat = textMat.resize(dSize, fx, fy)
    const ocrMat = prepForOCR(resizedMat, selectedImageFolder)
    // console.log('<<<RESIZED DIMENSIONS', resizedMat.cols, resizedMat.rows)
    const encodedBuffer = getTtfEncodedBuffer(ocrMat) // get 300 dpi encoded tiff buffer
    const { data: { text } } = await worker.recognize(encodedBuffer)
    cv.imwrite(getImagePath(selectedImageFolder, 'resized-text.png'), resizedMat)
    values.push(text);
  }
  console.log('Rect Text', values)
  await worker.terminate()
})()

function getHorizontalContours(inputMat, imageFolder) {
  const detectedHorizontalsMat = getHorizontalsMat(inputMat, imageFolder)
  return detectedHorizontalsMat.findContours(cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
}

function getHorizontalsMat(inputMat, imageFolder) {
  const pathTo = (imageName) => getImagePath(imageFolder, imageName)
  const horizontalKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(40, 1))
  const detectedHorizontalsMat = inputMat.morphologyEx(horizontalKernel, cv.MORPH_OPEN, new cv.Point(-1, -1), 2)
  cv.imwrite(pathTo('detectedHorizontals.png'), detectedHorizontalsMat)
  return detectedHorizontalsMat
}

function getVerticalContours(inputMat, imageFolder) {
  const detectedVerticalsMat = getVerticalsMat(inputMat, imageFolder)
  return detectedVerticalsMat.findContours(cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
}

function getVerticalsMat(inputMat, imageFolder) {
  const pathTo = (imageName) => getImagePath(imageFolder, imageName)
  const verticalKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, 10))
  const detectedVerticalsMat = inputMat.morphologyEx(verticalKernel, cv.MORPH_OPEN, new cv.Point(-1, -1), 2)
  cv.imwrite(pathTo('detectedVerticals.png'), detectedVerticalsMat)
  return detectedVerticalsMat
}

function getRects(mat, imageFolder) {
  const pathTo = (imageName) => getImagePath(imageFolder, imageName)
  const blurredMat = mat.gaussianBlur(new cv.Size(5, 5), 0)
  const edgedMat = blurredMat.canny(30, 200)
  cv.imwrite(pathTo('edged.png'), edgedMat)
  const cnts = edgedMat.findContours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  const rects = []

  for (const cnt of cnts) {
    const rect = cnt.boundingRect()
    if (filterRect(rect)) {
      rects.push(rect) 
    }
  }
  return rects.sort((rect1, rect2) => {
    if (rect1.y > rect2.y) {
      return 1
    } else if (rect1.y < rect2.y) {
      return -1
    } else if (rect1.x > rect2.x) {
      return 1
    } else if (rect1.x < rect2.x) {
      return -1
    } else {
      return 0
    }
  })
}

function getTextRects(mat, rects, imageFolder) {
  const pathTo = (imageName) => getImagePath(imageFolder, imageName)
  const textRects = []

  let index = 0
  for (const rect of rects) {
    // console.log(rect)
    const rectMat = mat.getRegion(rect)
    fillRectSides(rectMat, 4, 255)
    cv.imwrite(pathTo('textBox.png'), rectMat)
    const { height: lineHeightWithoutOffset, offset } = getTextHeight(rectMat, rect)
    const lineHeight = lineHeightWithoutOffset + offset
    const halfLineHeight = Math.floor((lineHeight) / 2)
    const erodeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(lineHeight, halfLineHeight))
    const errodedMat = rectMat.erode(erodeKernel, new cv.Point(lineHeight - 1, Math.floor(halfLineHeight / 2)), 2, cv.BORDER_ISOLATED)
    cv.imwrite(pathTo('erroded.png'), errodedMat)
    const binaryMat = errodedMat.threshold(127, 255, cv.THRESH_BINARY_INV)
    cv.imwrite(pathTo('binary.png'), binaryMat)
    const cnts = binaryMat.findContours(cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE, new cv.Point(0, 0))
    const boundingRects = cnts.map(cnt => cnt.boundingRect()).filter(bndRect => bndRect.width !== rect.width || bndRect.height !== rect.height)
    if (boundingRects.length > 1) {
      console.log(rect)
      console.log('erroded boundingRects', boundingRects)
    } else if (boundingRects.length === 1) {
      const { x, y } = rect
      const { width, height } = boundingRects.pop()
      // console.log(width, height)
      textRects.push({ rect: new cv.Rect(x, y, width, height), lineHeight: lineHeightWithoutOffset, lineOffset: offset })
    } else {
      console.log(rect)
      console.log('no bounding rects found')
    }

    if (selectedIndex !== undefined && selectedIndex === index) {
      break
    }
    index++
  }
  return textRects
}

function getTextHeight(mat, rect) {
  const blurredMat = mat.gaussianBlur(new cv.Size(5, 5), 0)
  const edgedMat = blurredMat.canny(30, 200)
  const cnts = edgedMat.findContours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  const boundingRects = cnts.map(cnt => cnt.boundingRect())
  const finalRect = boundingRects
    .filter(bndRect => bndRect.width !== rect.width || bndRect.height !== rect.height)
    .sort((rect1, rect2) => {
      if (rect1.height > rect2.height) {
        return 1
      } else if (rect1.height < rect2.height) {
        return -1
      } else if (rect1.y > rect2.y) {
        return 1
      } else if (rect.y < rect2.y) {
        return -1
      } else {
        return 0
      }
    })
    .pop()
  // console.log('<<<getTextHeight - finalRect', finalRect)
  return { height: finalRect.height, offset: finalRect.y }
}

function getRotationDegreesToLevelMat(inputMat, insets, imageFolder) {
  const mat = prepMatForRectDetection(inputMat, selectedImageFolder)
  const pathTo = (imageName) => getImagePath(imageFolder, imageName)
  let { rows, cols } = mat
  const [top, right, bottom, left] = insets
  const x = left
  const y = top
  const width = cols - left - right
  const height = (rows - bottom) - top
  const regionMat = mat.getRegion(new cv.Rect(x, y, width, height))
  cv.imwrite(pathTo('region/original.png'), regionMat)
  const cnts = getHorizontalContours(regionMat, imageFolder + '/region')
  const points = [...cnts[0].getPoints()]
  points.sort((p1, p2) => p1.x - p2.x)
  const { minY, maxY } = getAxesBoundsFromPoints(points)
  const a = maxY - minY
  const b = points.pop().x - points.shift().x
  const c = Math.sqrt(Math.pow(a, 2) + Math.pow(b, 2))
  const radians = Math.acos(b/c)
  const angle = toDegrees(radians)
  return angle / 2 // degree of angle overcompensates, so we half it
}

function prepMatForRectDetection(mat, imageFolder) {
  const pathTo = (imageName) => getImagePath(imageFolder, imageName)
  // cv.imwrite(pathTo('gray.png'), grayMat)
  const thresholdedMat = mat.threshold(0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
  cv.imwrite(pathTo('preppedForRectDetection.png'), thresholdedMat)
  return thresholdedMat
}

function prepForOCR(mat, imageFolder) {
  const pathTo = (imageName) => getImagePath(imageFolder, imageName)
  cv.imwrite(pathTo('ocr/original.png'), mat)
  // const blurredMat = mat.gaussianBlur(new cv.Size(5, 5), 0)
  //   cv.imwrite(pathTo('ocr/blurred.png'), blurredMat)

  const binaryMat = mat.threshold(190, 255, cv.THRESH_BINARY)
  cv.imwrite(pathTo('ocr/binary.png'), binaryMat)

  // const kernel = new cv.Mat(1,1, cv.CV_8UC1, 255)
  // const openMat = binaryMat.morphologyEx(kernel, cv.MORPH_OPEN)
  // cv.imwrite(pathTo('ocr/open.png'), openMat)
  // const errodedMat = binaryMat.erode(kernel, new cv.Point(-1, -1), 2)
  // cv.imwrite(pathTo('ocr/erroded.png'), errodedMat)
  // const closedMat = binaryMat.morphologyEx(kernel, cv.MORPH_CLOSE)
  // cv.imwrite(pathTo('ocr/closed.png'), closedMat)
  // const dilatedMat = binaryMat.dilate(kernel, new cv.Point(-1, -1), 4)
  // cv.imwrite(pathTo('ocr/dilated.png'), dilatedMat)
  const orMat = mat.bitwiseOr(binaryMat)
  cv.imwrite(pathTo('ocr/or.png'), orMat)
  return orMat
}

function fillRectSides(mat, thickness, value) {
  const { rows, cols } = mat
  const topSide = [
    new cv.Point(0, 0),
    new cv.Point(cols, thickness)
  ]
  const rightSide = [
    new cv.Point(cols - thickness, 0),
    new cv.Point(cols, rows),
  ]
  const bottomSide = [
    new cv.Point(0, rows - 5),
    new cv.Point(cols, rows) 
  ]
  const leftSide = [
    new cv.Point(0, 0),
    new cv.Point(thickness, rows)
  ]

  fillArea(mat, topSide, value)
  fillArea(mat, rightSide, value)
  fillArea(mat, bottomSide, value)
  fillArea(mat, leftSide, value)
}

function fillArea(mat, [startPoint, endPoint], value, condition) {
  // console.log(startPoint, endPoint)
  for (let col = startPoint.x; col < endPoint.x; col++) {
    for (let row = startPoint.y; row < endPoint.y; row++) {
      if (!condition || condition(mat.at(row, col))) {
        mat.set(row, col, value)
      }
    }
  }
}

function logContours(cnts) {
  let index = 0
  for (const cnt of cnts) {
    const points = cnt.getPoints()
    console.log('<<<CONTOUR ', index)
    console.log(cnt)
    console.log("points: ", points)
    index++
  }
}

function getAxesBoundsFromPoints(points) {
  let minY = -1, maxY = 0, minX = -1, maxX = 0
  points.forEach(point => {
    if (minY === -1 || point.y < minY) {
      minY = point.y
    } else if (point.y > maxY) {
      maxY = point.y
    }
    if (minX === -1 || point.x < minX) {
      minX = point.x
    } else if (point.x > maxX) {
      maxX = point.x
    }
  })
  return { minY, maxY, minX, maxX }
}

function getRectFromAxesBounds({ minY, maxY, minX, maxX}) {
  const height = maxY - minY
  const width = maxX - minX
  return new cv.Rect(minX, minY, width, height)
}

function filterRect(rect) {
  return (rect.height > (minRectHeight - 2) && rect.height < (minRectHeight + 6)) && rect.width > (minRectWidth - 6)
}

function getImagePath(imageFolder, imageName) {
  return `${imageFolder}/${imageName}`
}

function toDegrees (angle) {
  return angle * (180 / Math.PI)
}

function getTtfEncodedBuffer(mat) {
  const encodedBuffer = cv.imencode('.tiff', mat, [
    constants.IMWRITE_TIFF_RESUNIT, 2,
    constants.IMWRITE_TIFF_XDPI, 300,
    constants.IMWRITE_TIFF_YDPI, 300,
    constants.IMWRITE_TIFF_COMPRESSION, 1,
  ])
  return encodedBuffer
}
