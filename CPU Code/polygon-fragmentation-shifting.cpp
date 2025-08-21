#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sqlite3.h>

struct Point2D {
    float x;
    float y;
    
    Point2D(float x = 0, float y = 0) : x(x), y(y) {}
};

struct PolygonData {
    std::vector<Point2D> points;
    std::vector<int> polygonSizes;
};

// Function to read polygon data from SQLite database
PolygonData readDB(const std::string& dbFile) {
    sqlite3* db;
    if (sqlite3_open(dbFile.c_str(), &db) != SQLITE_OK) {
        throw std::runtime_error("Error opening database: " + dbFile);
    }

    PolygonData data;
    std::string query = "SELECT polygon_id, x, y FROM polygons ORDER BY polygon_id, id;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Error preparing query.");
    }

    int lastPolygonID = -1;
    int currentSize = 0;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int polygonID = sqlite3_column_int(stmt, 0);
        float x = static_cast<float>(sqlite3_column_double(stmt, 1));
        float y = static_cast<float>(sqlite3_column_double(stmt, 2));

        if (polygonID != lastPolygonID) {
            if (currentSize > 0) {
                data.polygonSizes.push_back(currentSize);
            }
            currentSize = 0;
            lastPolygonID = polygonID;
        }

        data.points.emplace_back(x, y);
        currentSize++;
    }

    if (currentSize > 0) {
        data.polygonSizes.push_back(currentSize);
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    
    return data;
}

PolygonData processAndShift(const PolygonData& input, float stepSize, float shiftDist) {
    PolygonData output;
    size_t total_points = 0;
    for (int polySize : input.polygonSizes) {
        total_points += polySize;
    }
    output.points.reserve(total_points);

    size_t currentIndex = 0;
    for (int polySize : input.polygonSizes) {
        std::vector<Point2D> newPoints;
        int shiftToggle = 0;

        std::vector<Point2D> originalPoints;
        for (int i = 0; i < polySize; ++i) {
            // Get the current point in the polygon
            const Point2D& current = input.points[currentIndex + i];
            // Get the next point in the polygon, wrapping around to the start if necessary
            const Point2D& next = input.points[currentIndex + ((i + 1) % polySize)];
            // Add the current point to the original points vector
            originalPoints.push_back(current);
            
            // Vertical Edge
            if (std::abs(current.x - next.x) < 1e-5f) {
                // moving up or down determines step direction
                float step = (next.y > current.y) ? stepSize : -stepSize;
                for (float y = current.y + step; std::abs(y - next.y) > 1e-5f; y += step) {
                    originalPoints.emplace_back(current.x, y); // add vertical points till edge-end
                }
            } 
            // Horizontal Edge
            else {
                // moving left or right determines step direction
                float step = (next.x > current.x) ? stepSize : -stepSize;
                for (float x = current.x + step; std::abs(x - next.x) > 1e-5f; x += step) {
                    originalPoints.emplace_back(x, current.y); // add horizontal points till edge-end
                }
            }
        }

        for (size_t i = 0; i < originalPoints.size(); ++i) { // traverse (modified) original points again for shifting
            size_t nextIdx = (i + 1) % originalPoints.size();
            const Point2D& current = originalPoints[i];
            const Point2D& next = originalPoints[nextIdx];

            newPoints.push_back(current); // first point of shifting

            Point2D shifted;
            if (shiftToggle % 2 == 0) {
                shifted = (std::abs(current.x - next.x) < 1e-5f) ? Point2D{current.x + shiftDist, current.y}
                                                                 : Point2D{current.x, current.y + shiftDist};
            } else {
                shifted = (std::abs(current.x - next.x) < 1e-5f) ? Point2D{current.x - shiftDist, current.y}
                                                                 : Point2D{current.x, current.y - shiftDist};
            }
            newPoints.push_back(shifted); // add back shifted points

            Point2D moved = (std::abs(current.x - next.x) < 1e-5f) ? Point2D{shifted.x, shifted.y + (next.y - current.y)}
                                                                   : Point2D{shifted.x + (next.x - current.x), shifted.y};
            newPoints.push_back(moved); // add back moved points

            shiftToggle++;
        }

        output.polygonSizes.push_back(newPoints.size());
        output.points.insert(output.points.end(), newPoints.begin(), newPoints.end());
        currentIndex += polySize;
    }

    return output;
}

// Function to write polygon data into SQLite database
void writeDB(const std::string& dbFile, const PolygonData& data) {
    sqlite3* db;
    if (sqlite3_open(dbFile.c_str(), &db) != SQLITE_OK) {
        throw std::runtime_error("Error opening database: " + dbFile);
    }

    std::string createTableQuery =
        "CREATE TABLE IF NOT EXISTS processed_polygons ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "polygon_id INTEGER, "
        "x REAL, "
        "y REAL);";
    
    char* errMsg = nullptr;
    if (sqlite3_exec(db, createTableQuery.c_str(), nullptr, nullptr, &errMsg) != SQLITE_OK) {
        sqlite3_close(db);
        throw std::runtime_error("Error creating table: " + std::string(errMsg));
    }

    std::string insertQuery = "INSERT INTO processed_polygons (polygon_id, x, y) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, insertQuery.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        throw std::runtime_error("Error preparing insert query.");
    }

    size_t currentIndex = 0;
    int polygonID = 0;
    for (int polySize : data.polygonSizes) {
        for (int i = 0; i < polySize; ++i) {
            const auto& p = data.points[currentIndex + i];

            sqlite3_bind_int(stmt, 1, polygonID);
            sqlite3_bind_double(stmt, 2, p.x);
            sqlite3_bind_double(stmt, 3, p.y);

            if (sqlite3_step(stmt) != SQLITE_DONE) {
                sqlite3_finalize(stmt);
                sqlite3_close(db);
                throw std::runtime_error("Error inserting polygon data.");
            }
            sqlite3_reset(stmt);
        }
        polygonID++;
        currentIndex += polySize;
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

int main() {
    try {
        const std::string dbFile = "DB Files/polygon.db";
        const float stepSize = 0.5f;
        const float shiftDist = 0.05f;

        PolygonData input = readDB(dbFile);
        if (input.points.empty()) {
            std::cerr << "No points read from database." << std::endl;
            return 1;
        }

        PolygonData output = processAndShift(input, stepSize, shiftDist);
        writeDB(dbFile, output);
        std::cout << "Processing complete. Results written to database." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}