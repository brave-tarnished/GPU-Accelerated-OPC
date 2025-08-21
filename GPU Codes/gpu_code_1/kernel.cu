#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sqlite3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <filesystem>
#include <regex>
#include <chrono>

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

    __host__ __device__ Point2D(float x = 0, float y = 0) : x(x), y(y) {}
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

// Utility function to measure and print execution time
class Timer {
private:
    std::string m_name;
    std::chrono::high_resolution_clock::time_point m_start;
    bool m_stopped;

public:
    Timer(const std::string& name) : m_name(name), m_stopped(false) {
        m_start = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        if (!m_stopped) {
            stop();
        }
    }

    void stop() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start).count();
        std::cout << m_name << " took " << duration << " ms" << std::endl;
        m_stopped = true;
    }
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
    // Use shared memory for temporary storage
    extern __shared__ Point2D sharedMem[];
    Point2D* fragmentedPoints = sharedMem;
    Point2D* newPoints = &sharedMem[1024];
    int fragmentedCount = 0;
    int newPointCount = 0;
    int shiftToggle = 0;

    // Store the first point to ensure proper closure
    Point2D firstPoint = originalPoints[0];

    // Generate fragment points - using a proper loop structure to ensure closure
    // Loop from 0 to polySize-1 to avoid processing the closing edge twice
    for (int i = 0; i < polySize; ++i) {
        const Point2D& current = originalPoints[i];
        const Point2D& next = originalPoints[(i + 1) % polySize];
        const Point2D& prev = originalPoints[(i - 1 + polySize) % polySize];

        // Add the current point to fragmented points
        fragmentedPoints[fragmentedCount++] = current;

        bool currentIsCorner = isCorner(prev, current, next);
        bool nextIsCorner = isCorner(current, next, originalPoints[(i + 2) % polySize]);

        float stepSize = edgeStepSize;

        // Existing edge fragmentation logic...
        // [Keep the existing vertical and horizontal edge processing code]
    }

    // Apply shifting logic to fragmented points
    // Make sure we don't create unwanted connections
    for (int i = 0; i < fragmentedCount; ++i) {
        int nextIdx = (i + 1) % fragmentedCount;
        const Point2D& current = fragmentedPoints[i];
        const Point2D& next = fragmentedPoints[nextIdx];

        // Add the current point
        newPoints[newPointCount++] = current;

        // Skip adding shifted points for the last point if it would create an unwanted connection
        // This is crucial to prevent the star pattern
        if (i == fragmentedCount - 1 &&
            fabsf(current.x - fragmentedPoints[0].x) < 1e-5f &&
            fabsf(current.y - fragmentedPoints[0].y) < 1e-5f) {
            // We're at the last point and it matches the first point
            // Skip adding shifted points to avoid creating unwanted connections
            continue;
        }

        // Add shifted points
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

    // Ensure the polygon is properly closed
    // Check if the first and last points match
    if (newPointCount > 0 &&
        (fabsf(newPoints[0].x - newPoints[newPointCount - 1].x) > 1e-5f ||
            fabsf(newPoints[0].y - newPoints[newPointCount - 1].y) > 1e-5f)) {
        // The polygon is not closed - add the first point at the end
        newPoints[newPointCount++] = newPoints[0];
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
        deviceData.numPolygons * sizeof(int), cudaMemcpyDeviceToHost));

    // Calculate total number of points
    int totalPoints = 0;
    for (int size : hostSizes) {
        totalPoints += size;
    }

    // Allocate memory for points on host
    hostData.points.resize(totalPoints);
    hostData.polygonSizes = hostSizes;

    // Copy points back to host
    CUDA_CHECK(cudaMemcpy(hostData.points.data(), deviceData.points,
        totalPoints * sizeof(Point2D), cudaMemcpyDeviceToHost));

    return hostData;
}



int main(int argc, char** argv) {
    try {
        Timer totalTimer("Total program execution");

        // Parameters for polygon processing
        float edgeStepSize = 0.5f;
        float cornerStepSize = 0.1f;
        int horizontalCornerFragments = 2;
        int verticalCornerFragments = 3;
        float shiftDist = 0.05f;

        // Directory containing DB files
        std::string dbDirectory = "D:/Academics/Semester VI/EE 399 - CUDA Project/POST-MIDSEM/DB Files/Divided/";

        // Get list of DB files to process
        std::vector<std::string> dbFiles;
        // In a real implementation, you would scan the directory for .db files
        // For simplicity, we'll add files manually
        dbFiles.push_back(dbDirectory + "small_layout_block_0_0.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_0_1.db");/*
        dbFiles.push_back(dbDirectory + "small_layout_block_0_2.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_0_3.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_0_4.db");*/
        dbFiles.push_back(dbDirectory + "small_layout_block_1_0.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_1_1.db");/*
        dbFiles.push_back(dbDirectory + "small_layout_block_1_2.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_1_3.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_1_4.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_2_0.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_2_1.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_2_2.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_2_3.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_2_4.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_3_0.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_3_1.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_3_2.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_3_3.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_3_4.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_4_0.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_4_1.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_4_2.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_4_3.db");
        dbFiles.push_back(dbDirectory + "small_layout_block_4_4.db");*/

        std::cout << "Processing " << dbFiles.size() << " DB files using CUDA" << std::endl;

        // Process each DB file
        for (const auto& dbFile : dbFiles) {
            std::cout << "Processing file: " << dbFile << std::endl;
            Timer fileTimer("File processing: " + dbFile);

            // Read input data from database
            {
                Timer readTimer("Database reading");
                PolygonData inputData = readDB(dbFile);
                if (inputData.points.empty()) {
                    std::cerr << "No points read from database: " << dbFile << std::endl;
                    continue;
                }

                std::cout << "Read " << inputData.points.size() << " points in "
                    << inputData.polygonSizes.size() << " polygons" << std::endl;

                // Prepare data for GPU
                Timer gpuPrepTimer("GPU data preparation");
                DevicePolygonData d_inputData = prepareInputData(inputData);
                DevicePolygonData d_outputData = prepareOutputData(d_inputData);
                gpuPrepTimer.stop();

                // Calculate kernel launch parameters
                int threadsPerBlock = 256;
                int numBlocks = (d_inputData.numPolygons + threadsPerBlock - 1) / threadsPerBlock;

                // Calculate shared memory size - each polygon needs space for fragmented and new points
                size_t sharedMemSize = 2 * 1024 * sizeof(Point2D); // Space for 1024 fragmented points + 1024 new points

                std::cout << "Launching kernel with " << numBlocks << " blocks, "
                    << threadsPerBlock << " threads per block" << std::endl;

                // Launch kernel and time it
                {
                    Timer kernelTimer("CUDA kernel execution");
                    processPolygonsKernel << <numBlocks, threadsPerBlock, sharedMemSize >> > (
                        d_inputData, d_outputData,
                        edgeStepSize, cornerStepSize,
                        horizontalCornerFragments, verticalCornerFragments,
                        shiftDist
                        );

                    // Check for kernel launch errors
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
                        freeDeviceData(d_inputData);
                        freeDeviceData(d_outputData);
                        continue;
                    }

                    // Wait for GPU to finish
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

                // Retrieve results from GPU
                Timer retrieveTimer("Result retrieval from GPU");
                PolygonData outputData = retrieveResults(d_outputData);
                retrieveTimer.stop();

                std::cout << "Processed data: " << outputData.points.size() << " points in "
                    << outputData.polygonSizes.size() << " polygons" << std::endl;

                // Write results back to database
                Timer writeTimer("Database writing");
                writeDB(dbFile, outputData);
                writeTimer.stop();

                std::cout << "Results written to database: " << dbFile << std::endl;

                // Free GPU memory
                Timer cleanupTimer("GPU memory cleanup");
                freeDeviceData(d_inputData);
                freeDeviceData(d_outputData);
                cleanupTimer.stop();
            }

            fileTimer.stop();
        }

        std::cout << "All files processed successfully." << std::endl;
        totalTimer.stop();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
