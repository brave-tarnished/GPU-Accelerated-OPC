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

bool isCorner(const Point2D& prev, const Point2D& current, const Point2D& next) {
    bool prevVertical = std::abs(current.x - prev.x) < 1e-5f;
    bool nextVertical = std::abs(next.x - current.x) < 1e-5f;
    return prevVertical != nextVertical;
}

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


PolygonData processAndShift(const PolygonData& input, float edgeStepSize, float cornerStepSize, 
                          int horizontalCornerFragments, int verticalCornerFragments, float shiftDist) {
    PolygonData output;
    size_t total_points = 0;
    for (int polySize : input.polygonSizes) {
        total_points += polySize;
    }
    output.points.reserve(total_points);
    
    size_t currentIndex = 0;
    for (int polySize : input.polygonSizes) {
        std::vector<Point2D> fragmentedPoints;
        std::vector<Point2D> originalPoints;
        std::vector<Point2D> newPoints;
        int shiftToggle = 0;

        // First, collect original polygon points
        for (int i = 0; i < polySize; ++i) {
            originalPoints.push_back(input.points[currentIndex + i]);
        }
        
        // Generate fragment points
        for (int i = 0; i < polySize; ++i) {
            const Point2D& current = originalPoints[i];
            const Point2D& next = originalPoints[(i + 1) % polySize];
            const Point2D& prev = originalPoints[(i - 1 + polySize) % polySize];
            
            fragmentedPoints.push_back(current);
            
            bool currentIsCorner = isCorner(prev, current, next);
            bool nextIsCorner = isCorner(current, next, originalPoints[(i + 2) % polySize]);
            
            float stepSize = edgeStepSize;
            
            if (std::abs(current.x - next.x) < 1e-5f) {
                // Vertical edge
                float direction = (next.y > current.y) ? 1.0f : -1.0f;
                
                if (currentIsCorner) {
                    for (int j = 1; j <= verticalCornerFragments; ++j) {
                        float y = current.y + direction * j * cornerStepSize;
                        if ((direction > 0 && y < next.y) || (direction < 0 && y > next.y)) {
                            fragmentedPoints.emplace_back(current.x, y);
                        }
                    }
                }
                
                float startPos = current.y + (currentIsCorner ? direction * verticalCornerFragments * cornerStepSize : 0);
                float endPos = next.y - (nextIsCorner ? direction * verticalCornerFragments * cornerStepSize : 0);
                
                if (direction > 0) {
                    for (float y = std::max(startPos, current.y) + stepSize; y < std::min(endPos, next.y); y += stepSize) {
                        fragmentedPoints.emplace_back(current.x, y);
                    }
                } else {
                    for (float y = std::min(startPos, current.y) - stepSize; y > std::max(endPos, next.y); y -= stepSize) {
                        fragmentedPoints.emplace_back(current.x, y);
                    }
                }
                
                if (nextIsCorner) {
                    for (int j = verticalCornerFragments; j >= 1; --j) {
                        float y = next.y - direction * j * cornerStepSize;
                        if ((direction > 0 && y < next.y && y > current.y) || (direction < 0 && y > next.y && y < current.y)) {
                            fragmentedPoints.emplace_back(current.x, y);
                        }
                    }
                }
            } else {
                // Horizontal edge
                float direction = (next.x > current.x) ? 1.0f : -1.0f;
                
                if (currentIsCorner) {
                    for (int j = 1; j <= horizontalCornerFragments; ++j) {
                        float x = current.x + direction * j * cornerStepSize;
                        if ((direction > 0 && x < next.x) || (direction < 0 && x > next.x)) {
                            fragmentedPoints.emplace_back(x, current.y);
                        }
                    }
                }
                
                float startPos = current.x + (currentIsCorner ? direction * horizontalCornerFragments * cornerStepSize : 0);
                float endPos = next.x - (nextIsCorner ? direction * horizontalCornerFragments * cornerStepSize : 0);
                
                if (direction > 0) {
                    for (float x = std::max(startPos, current.x) + stepSize; x < std::min(endPos, next.x); x += stepSize) {
                        fragmentedPoints.emplace_back(x, current.y);
                    }
                } else {
                    for (float x = std::min(startPos, current.x) - stepSize; x > std::max(endPos, next.x); x -= stepSize) {
                        fragmentedPoints.emplace_back(x, current.y);
                    }
                }
                
                if (nextIsCorner) {
                    for (int j = horizontalCornerFragments; j >= 1; --j) {
                        float x = next.x - direction * j * cornerStepSize;
                        if ((direction > 0 && x < next.x && x > current.x) || (direction < 0 && x > next.x && x < current.x)) {
                            fragmentedPoints.emplace_back(x, current.y);
                        }
                    }
                }
            }
        }
        
        // Now apply the shifting logic to the fragmented points
        for (size_t i = 0; i < fragmentedPoints.size(); ++i) {
            size_t nextIdx = (i + 1) % fragmentedPoints.size();
            const Point2D& current = fragmentedPoints[i];
            const Point2D& next = fragmentedPoints[nextIdx];

            newPoints.push_back(current);

            Point2D shifted;
            if (shiftToggle % 2 == 0) {
                shifted = (std::abs(current.x - next.x) < 1e-5f) ? Point2D{current.x + shiftDist, current.y}
                                                                 : Point2D{current.x, current.y + shiftDist};
            } else {
                shifted = (std::abs(current.x - next.x) < 1e-5f) ? Point2D{current.x - shiftDist, current.y}
                                                                 : Point2D{current.x, current.y - shiftDist};
            }
            newPoints.push_back(shifted);

            Point2D moved = (std::abs(current.x - next.x) < 1e-5f) ? Point2D{shifted.x, shifted.y + (next.y - current.y)}
                                                                   : Point2D{shifted.x + (next.x - current.x), shifted.y};
            newPoints.push_back(moved);

            shiftToggle++;
        }

        output.polygonSizes.push_back(newPoints.size());
        output.points.insert(output.points.end(), newPoints.begin(), newPoints.end());
        currentIndex += polySize;
    }

    return output;
}



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
        const float edgeStepSize = 1.0f;
        const float cornerStepSize = 0.1f;
        const int horizontalCornerFragments = 2;
        const int verticalCornerFragments = 3;
        const float shiftDist = 0.05f;

        PolygonData input = readDB(dbFile);
        if (input.points.empty()) {
            std::cerr << "No points read from database." << std::endl;
            return 1;
        }

        PolygonData output = processAndShift(input, edgeStepSize, cornerStepSize, 
                                           horizontalCornerFragments, verticalCornerFragments,
                                           shiftDist);
        writeDB(dbFile, output);
        std::cout << "Processing complete. Results written to database." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}