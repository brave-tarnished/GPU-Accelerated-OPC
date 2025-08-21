#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cerrno>
#include <algorithm>

struct Point2D {
    float x;
    float y;
    
    Point2D(float x = 0, float y = 0) : x(x), y(y) {}
};

struct PolygonData {
    std::vector<Point2D> points;
    std::vector<int> polygonSizes;
};

PolygonData readCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    PolygonData data;
    std::vector<Point2D> currentPolygon;
    std::string line;

    while (std::getline(file, line)) {
        // Fixed line: Use erase-remove idiom correctly
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        
        if (line.find("POLYGON_END") != std::string::npos) {
            if (!currentPolygon.empty()) {
                data.polygonSizes.push_back(static_cast<int>(currentPolygon.size()));
                data.points.insert(data.points.end(), currentPolygon.begin(), currentPolygon.end());
                currentPolygon.clear();
            }
            continue;
        }

        size_t commaPos = line.find(',');
        if (commaPos != std::string::npos) {
            try {
                float x = std::stof(line.substr(0, commaPos));
                float y = std::stof(line.substr(commaPos + 1));
                currentPolygon.emplace_back(x, y);
            } catch (...) {
                // Ignore malformed lines
            }
        }
    }

    return data;
}

void addIntermediatePoints(const Point2D& p1, const Point2D& p2, 
                          std::vector<Point2D>& result, float stepSize) {
    result.push_back(p1);

    if (std::abs(p1.x - p2.x) < 1e-5f) {  // Vertical edge
        float startY = std::min(p1.y, p2.y);
        float endY = std::max(p1.y, p2.y);

        for (float y = startY + stepSize; y < endY; y += stepSize) {
            result.emplace_back(p1.x, y);
        }
    }
    else if (std::abs(p1.y - p2.y) < 1e-5f) {  // Horizontal edge
        float startX = std::min(p1.x, p2.x);
        float endX = std::max(p1.x, p2.x);

        for (float x = startX + stepSize; x < endX; x += stepSize) {
            result.emplace_back(x, p1.y);
        }
    }
}

PolygonData processPolygons(const PolygonData& input, float stepSize) {
    PolygonData output;
    output.polygonSizes.reserve(input.polygonSizes.size());

    size_t currentIndex = 0;
    for (int polySize : input.polygonSizes) {
        std::vector<Point2D> processedPolygon;
        processedPolygon.reserve(polySize * (10 / stepSize + 1));

        for (int i = 0; i < polySize; i++) {
            const Point2D& current = input.points[currentIndex + i];
            const Point2D& next = input.points[currentIndex + ((i + 1) % polySize)];
            addIntermediatePoints(current, next, processedPolygon, stepSize);
        }

        output.polygonSizes.push_back(static_cast<int>(processedPolygon.size()));
        output.points.insert(output.points.end(), processedPolygon.begin(), processedPolygon.end());
        currentIndex += polySize;
    }

    return output;
}

void writeCSV(const std::string& filename, const PolygonData& data) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Error opening output file: " + filename);
    }

    size_t currentIndex = 0;
    for (int polySize : data.polygonSizes) {
        for (int i = 0; i < polySize; i++) {
            const auto& p = data.points[currentIndex + i];
            file << p.x << "," << p.y << "\n";
        }
        file << "POLYGON_END,POLYGON_END\n";
        currentIndex += polySize;
    }
}

int main() {
    try {
        const std::string inputFile = "input.csv";
        const std::string outputFile = "fragmented_output.csv";
        const float stepSize = 0.5f;

        std::cout << "Processing with step size: " << stepSize << std::endl;

        PolygonData input = readCSV(inputFile);
        if (input.points.empty()) {
            std::cerr << "No points read from input file." << std::endl;
            return 1;
        }

        std::cout << "Read " << input.points.size() << " points across " 
                  << input.polygonSizes.size() << " polygons" << std::endl;

        PolygonData output = processPolygons(input, stepSize);
        std::cout << "Generated " << output.points.size() << " output points" << std::endl;

        writeCSV(outputFile, output);
        std::cout << "Processing complete. Results written to " << outputFile << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}