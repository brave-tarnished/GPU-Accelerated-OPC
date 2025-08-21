#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sqlite3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Point structure for both host and device
struct Point2D {
    float x;
    float y;

    Point2D(float x = 0, float y = 0) : x(x), y(y) {}
};

// Polygon data structure for host
struct PolygonData {
    std::vector<Point2D> points;
    std::vector<int> polygonSizes;
};

// Device-side polygon data structure
struct DevicePolygonData {
    Point2D* points;
    int* polygonSizes;
    int* polygonOffsets;  // Starting index of each polygon in the points array
    int numPolygons;
    int totalPoints;
};



// Device function to check if a point is a corner
__device__ bool isCorner(const Point2D& prev, const Point2D& current, const Point2D& next) {
    bool prevVertical = fabsf(current.x - prev.x) < 1e-5f;
    bool nextVertical = fabsf(next.x - current.x) < 1e-5f;
    return prevVertical != nextVertical;
}

// Device function to process a single polygon
__device__ void processPolygon(const Point2D* originalPoints, int polySize,
    Point2D* outputPoints, int* outputSize,
    float edgeStepSize, float cornerStepSize,
    int horizontalCornerFragments, int verticalCornerFragments,
    float shiftDist) {
    // Temporary arrays for fragmented and new points
    // Using dynamic shared memory would be ideal, but for simplicity we'll use fixed sizes
    Point2D fragmentedPoints[1024];  // Adjust size based on expected maximum
    Point2D newPoints[3072];         // Each fragmented point can generate up to 3 new points
    int fragmentedCount = 0;
    int newPointCount = 0;
    int shiftToggle = 0;

    // Generate fragment points
    for (int i = 0; i < polySize; ++i) {
        const Point2D& current = originalPoints[i];
        const Point2D& next = originalPoints[(i + 1) % polySize];
        const Point2D& prev = originalPoints[(i - 1 + polySize) % polySize];

        fragmentedPoints[fragmentedCount++] = current;

        bool currentIsCorner = isCorner(prev, current, next);
        bool nextIsCorner = isCorner(current, next, originalPoints[(i + 2) % polySize]);

        float stepSize = edgeStepSize;

        if (fabsf(current.x - next.x) < 1e-5f) {
            // Vertical edge
            float direction = (next.y > current.y) ? 1.0f : -1.0f;

            if (currentIsCorner) {
                for (int j = 1; j <= verticalCornerFragments; ++j) {
                    float y = current.y + direction * j * cornerStepSize;
                    if ((direction > 0 && y < next.y) || (direction < 0 && y > next.y)) {
                        fragmentedPoints[fragmentedCount++] = { current.x, y };
                    }
                }
            }

            float startPos = current.y + (currentIsCorner ? direction * verticalCornerFragments * cornerStepSize : 0);
            float endPos = next.y - (nextIsCorner ? direction * verticalCornerFragments * cornerStepSize : 0);

            if (direction > 0) {
                for (float y = fmaxf(startPos, current.y) + stepSize; y < fminf(endPos, next.y); y += stepSize) {
                    fragmentedPoints[fragmentedCount++] = { current.x, y };
                }
            }
            else {
                for (float y = fminf(startPos, current.y) - stepSize; y > fmaxf(endPos, next.y); y -= stepSize) {
                    fragmentedPoints[fragmentedCount++] = { current.x, y };
                }
            }

            if (nextIsCorner) {
                for (int j = verticalCornerFragments; j >= 1; --j) {
                    float y = next.y - direction * j * cornerStepSize;
                    if ((direction > 0 && y < next.y && y > current.y) ||
                        (direction < 0 && y > next.y && y < current.y)) {
                        fragmentedPoints[fragmentedCount++] = { current.x, y };
                    }
                }
            }
        }
        else {
            // Horizontal edge
            float direction = (next.x > current.x) ? 1.0f : -1.0f;

            if (currentIsCorner) {
                for (int j = 1; j <= horizontalCornerFragments; ++j) {
                    float x = current.x + direction * j * cornerStepSize;
                    if ((direction > 0 && x < next.x) || (direction < 0 && x > next.x)) {
                        fragmentedPoints[fragmentedCount++] = { x, current.y };
                    }
                }
            }

            float startPos = current.x + (currentIsCorner ? direction * horizontalCornerFragments * cornerStepSize : 0);
            float endPos = next.x - (nextIsCorner ? direction * horizontalCornerFragments * cornerStepSize : 0);

            if (direction > 0) {
                for (float x = fmaxf(startPos, current.x) + stepSize; x < fminf(endPos, next.x); x += stepSize) {
                    fragmentedPoints[fragmentedCount++] = { x, current.y };
                }
            }
            else {
                for (float x = fminf(startPos, current.x) - stepSize; x > fmaxf(endPos, next.x); x -= stepSize) {
                    fragmentedPoints[fragmentedCount++] = { x, current.y };
                }
            }

            if (nextIsCorner) {
                for (int j = horizontalCornerFragments; j >= 1; --j) {
                    float x = next.x - direction * j * cornerStepSize;
                    if ((direction > 0 && x < next.x && x > current.x) ||
                        (direction < 0 && x > next.x && x < current.x)) {
                        fragmentedPoints[fragmentedCount++] = { x, current.y };
                    }
                }
            }
        }
    }

    // Apply shifting logic to fragmented points
    for (int i = 0; i < fragmentedCount; ++i) {
        int nextIdx = (i + 1) % fragmentedCount;
        const Point2D& current = fragmentedPoints[i];
        const Point2D& next = fragmentedPoints[nextIdx];

        newPoints[newPointCount++] = current;

        Point2D shifted;
        if (shiftToggle % 2 == 0) {
            shifted = (fabsf(current.x - next.x) < 1e-5f) ?
                Point2D{ current.x + shiftDist, current.y } :
                Point2D{ current.x, current.y + shiftDist };
        }
        else {
            shifted = (fabsf(current.x - next.x) < 1e-5f) ?
                Point2D{ current.x - shiftDist, current.y } :
                Point2D{ current.x, current.y - shiftDist };
        }
        newPoints[newPointCount++] = shifted;

        Point2D moved = (fabsf(current.x - next.x) < 1e-5f) ?
            Point2D{ shifted.x, shifted.y + (next.y - current.y) } :
            Point2D{ shifted.x + (next.x - current.x), shifted.y };
        newPoints[newPointCount++] = moved;

        shiftToggle++;
    }

    // Copy new points to output
    for (int i = 0; i < newPointCount; ++i) {
        outputPoints[i] = newPoints[i];
    }

    *outputSize = newPointCount;
}



__global__ void processPolygonsKernel(DevicePolygonData input, DevicePolygonData output,
    float edgeStepSize, float cornerStepSize,
    int horizontalCornerFragments, int verticalCornerFragments,
    float shiftDist) {
    int polyIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (polyIdx < input.numPolygons) {
        int startIdx = input.polygonOffsets[polyIdx];
        int polySize = input.polygonSizes[polyIdx];

        // Calculate output offset
        int outputOffset = output.polygonOffsets[polyIdx];

        // Process this polygon
        processPolygon(
            &input.points[startIdx],
            polySize,
            &output.points[outputOffset],
            &output.polygonSizes[polyIdx],
            edgeStepSize, cornerStepSize,
            horizontalCornerFragments, verticalCornerFragments,
            shiftDist
        );
    }
}



// Read polygon data from SQLite database
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

// Write polygon data to SQLite database
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

    // Begin transaction for better performance
    sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);

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

    // Commit transaction
    sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr);

    sqlite3_close(db);
}

// Prepare input data for GPU processing
DevicePolygonData prepareInputData(const PolygonData& hostData) {
    DevicePolygonData deviceData;
    deviceData.numPolygons = hostData.polygonSizes.size();
    deviceData.totalPoints = hostData.points.size();

    // Allocate device memory for points
    CUDA_CHECK(cudaMalloc(&deviceData.points, deviceData.totalPoints * sizeof(Point2D)));
    CUDA_CHECK(cudaMemcpy(deviceData.points, hostData.points.data(),
        deviceData.totalPoints * sizeof(Point2D), cudaMemcpyHostToDevice));

    // Allocate device memory for polygon sizes
    CUDA_CHECK(cudaMalloc(&deviceData.polygonSizes, deviceData.numPolygons * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(deviceData.polygonSizes, hostData.polygonSizes.data(),
        deviceData.numPolygons * sizeof(int), cudaMemcpyHostToDevice));

    // Calculate and allocate polygon offsets
    std::vector<int> hostOffsets(deviceData.numPolygons);
    int offset = 0;
    for (int i = 0; i < deviceData.numPolygons; ++i) {
        hostOffsets[i] = offset;
        offset += hostData.polygonSizes[i];
    }

    CUDA_CHECK(cudaMalloc(&deviceData.polygonOffsets, deviceData.numPolygons * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(deviceData.polygonOffsets, hostOffsets.data(),
        deviceData.numPolygons * sizeof(int), cudaMemcpyHostToDevice));

    return deviceData;
}

// Prepare output data structure on GPU
DevicePolygonData prepareOutputData(const DevicePolygonData& inputData) {
    DevicePolygonData outputData;
    outputData.numPolygons = inputData.numPolygons;

    // Allocate memory for polygon sizes (will be filled by kernel)
    CUDA_CHECK(cudaMalloc(&outputData.polygonSizes, outputData.numPolygons * sizeof(int)));

    // Initialize with zeros
    CUDA_CHECK(cudaMemset(outputData.polygonSizes, 0, outputData.numPolygons * sizeof(int)));

    // First run to calculate output sizes
    // This is a simplified approach - in a real implementation, you might want to
    // do a separate kernel launch to calculate sizes first

    // For now, we'll estimate the maximum possible output size
    // Each point can generate up to 3 new points after fragmentation and shifting
    int estimatedMaxPointsPerPolygon = 3 * 1024;  // Assuming max 1024 points per polygon after fragmentation

    // Allocate memory for polygon offsets
    CUDA_CHECK(cudaMalloc(&outputData.polygonOffsets, outputData.numPolygons * sizeof(int)));

    // Initialize with conservative estimates
    std::vector<int> estimatedOffsets(outputData.numPolygons);
    int estimatedOffset = 0;
    for (int i = 0; i < outputData.numPolygons; ++i) {
        estimatedOffsets[i] = estimatedOffset;
        estimatedOffset += estimatedMaxPointsPerPolygon;
    }

    CUDA_CHECK(cudaMemcpy(outputData.polygonOffsets, estimatedOffsets.data(),
        outputData.numPolygons * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate memory for points (with conservative estimate)
    outputData.totalPoints = estimatedOffset;
    CUDA_CHECK(cudaMalloc(&outputData.points, outputData.totalPoints * sizeof(Point2D)));

    return outputData;
}

// Free device memory
void freeDeviceData(DevicePolygonData& data) {
    if (data.points) CUDA_CHECK(cudaFree(data.points));
    if (data.polygonSizes) CUDA_CHECK(cudaFree(data.polygonSizes));
    if (data.polygonOffsets) CUDA_CHECK(cudaFree(data.polygonOffsets));

    data.points = nullptr;
    data.polygonSizes = nullptr;
    data.polygonOffsets = nullptr;
}

// Retrieve results from GPU
PolygonData retrieveResults(const DevicePolygonData& deviceData) {
    PolygonData hostData;

    // Copy polygon sizes back to host
    std::vector<int> hostSizes(deviceData.numPolygons);
    CUDA_CHECK(cudaMemcpy(hostSizes.data(), deviceData.polygonSizes,
