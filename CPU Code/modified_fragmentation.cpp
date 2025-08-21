#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sqlite3.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

PolygonData processAndShift(const PolygonData& input, float baseStepSize, float shiftDist, float cornerThreshold = 0.1f) {
    PolygonData output;
    size_t total_points = 0;
    for (int polySize : input.polygonSizes) {
        total_points += polySize;
    }
    output.points.reserve(total_points * 2); // Rough estimate for added points

    size_t currentIndex = 0;
    for (int polySize : input.polygonSizes) {
        std::vector<Point2D> newPoints;

        // Store original points and calculate angles
        std::vector<Point2D> originalPoints(polySize);
        std::vector<float> angles(polySize);
        
        for (int i = 0; i < polySize; ++i) {
            originalPoints[i] = input.points[currentIndex + i];
            
            // Calculate angle at each vertex
            int prevIdx = (i - 1 + polySize) % polySize;
            int nextIdx = (i + 1) % polySize;
            
            Point2D prev = input.points[currentIndex + prevIdx];
            Point2D curr = input.points[currentIndex + i];
            Point2D next = input.points[currentIndex + nextIdx];
            
            float dx1 = curr.x - prev.x;
            float dy1 = curr.y - prev.y;
            float dx2 = next.x - curr.x;
            float dy2 = next.y - curr.y;
            
            float dot = dx1 * dx2 + dy1 * dy2;
            float mag1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
            float mag2 = std::sqrt(dx2 * dx2 + dy2 * dy2);
            
            // Avoid division by zero
            if (mag1 < 1e-5f || mag2 < 1e-5f) {
                angles[i] = 0.0f;
                continue;
            }
            
            float cosAngle = dot / (mag1 * mag2);
            // Clamp cosAngle to [-1, 1] to avoid numerical errors
            cosAngle = std::max(-1.0f, std::min(1.0f, cosAngle));
            float angle = std::acos(cosAngle);
            angles[i] = std::abs(M_PI - angle); // Difference from straight line
        }

        // Process each segment with adaptive step size
        for (int i = 0; i < polySize; ++i) {
            const Point2D& current = originalPoints[i];
            const Point2D& next = originalPoints[(i + 1) % polySize];
            
            // Determine step size based on corner proximity
            float cornerFactor = std::max(angles[i], angles[(i + 1) % polySize]);
            float stepSize = baseStepSize;
            if (cornerFactor > cornerThreshold) {
                // Smaller steps near corners (sharper angles)
                stepSize = baseStepSize / (1.0f + 3.0f * cornerFactor);
            }
            
            // Generate intermediate points
            newPoints.push_back(current); // Always include the starting point
            float dx = next.x - current.x;
            float dy = next.y - current.y;
            float length = std::sqrt(dx * dx + dy * dy);
            
            if (length < 1e-5f) {
                continue; // Skip if points are too close
            }
            
            int numSteps = std::max(1, static_cast<int>(length / stepSize));
            float actualStepX = dx / numSteps;
            float actualStepY = dy / numSteps;

            for (int step = 1; step < numSteps; ++step) {
                newPoints.emplace_back(
                    current.x + step * actualStepX,
                    current.y + step * actualStepY
                );
            }
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
        const std::string dbFile = "polygon.db";
        const float stepSize = 0.5f;
        const float shiftDist = 0.0f;
        const float cornerThreshold = 0.1f; // Angle threshold in radians (~5.7 degrees)

        PolygonData input = readDB(dbFile);
        if (input.points.empty()) {
            std::cerr << "No points read from database." << std::endl;
            return 1;
        }

        PolygonData output = processAndShift(input, stepSize, shiftDist, cornerThreshold);
        writeDB(dbFile, output);
        std::cout << "Processing complete. Results written to database." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}