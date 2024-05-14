#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "equality_assignment.h"
#include "greedy_by_breadth_assignment.h"
#include "greedy_by_size_assignment.h"
#include "greedy_in_order_assignment.h"
#include "min_cost_flow_assignment.h"
#include "naive_assignment.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management.h"
#include "types.h"

void StoreToCsv(
    tflite::gpu::OffsetsAssignment& offsets,
    tflite::gpu::ObjectsAssignment<size_t>& assignment,
    std::vector<tflite::gpu::TensorUsageRecord<size_t>>& usage_records,
    std::string& algorithm) {
  const char* path = std::getenv("TFLITE_CSV_PATH");
  const char* name = std::getenv("TFLITE_TRACE_NAME");

  if (path && name) {
    std::string path_string = std::string(path);
    std::string filename = std::string(name) + "-in.csv";

    std::string new_path =
        path_string + "/" + "tflite-csv-in" + "/" + algorithm + "/";
    std::ofstream outfile_in(new_path + filename, std::ios::trunc);

    if (outfile_in.is_open()) {
      outfile_in << "id,lower,upper,size" << std::endl;
      for (size_t i = 0; i < assignment.object_ids.size(); ++i) {
        outfile_in << i << "," << usage_records[i].first_task << ","
                   << usage_records[i].last_task + 1 << ","
                   << usage_records[i].tensor_size << std::endl;
      }
      outfile_in.close();
    } else {
      std::cout << "Could not open file: " << new_path << filename << std::endl;
    }

    filename = std::string(name) + "-out.csv";
    new_path = path_string + "/" + "tflite-csv-out" + "/" + algorithm + "/";
    std::ofstream outfile_out(new_path + filename, std::ios::trunc);

    if (outfile_out.is_open()) {
      outfile_out << "id,lower,upper,size,offset" << std::endl;
      for (size_t i = 0; i < assignment.object_ids.size(); ++i) {
        outfile_out << i << "," << usage_records[i].first_task << ","
                    << usage_records[i].last_task + 1 << ","
                    << usage_records[i].tensor_size << "," << offsets.offsets[i]
                    << std::endl;
      }
      outfile_out.close();
    } else {
      std::cout << "Could not open file: " << new_path << filename << std::endl;
    }
  } else {
    std::cout << "One or both environment variables not found." << std::endl;
  }
}

void fillTensorUsageRecord(
    const std::string& filepath,
    std::vector<tflite::gpu::TensorUsageRecord<size_t>>* usage_records) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filepath << std::endl;
    return;
  }

  std::string header;
  getline(file, header);

  std::string line;
  while (getline(file, line)) {
    std::stringstream ss(line);
    std::string value;

    // skip id
    getline(ss, value, ',');

    getline(ss, value, ',');
    auto lower = static_cast<size_t>(stoi(value));

    getline(ss, value, ',');
    auto upper = static_cast<size_t>(stoi(value)) - 1;

    getline(ss, value, ',');
    auto size = static_cast<size_t>(stoi(value));

    tflite::gpu::TensorUsageRecord<size_t> record =
        tflite::gpu::TensorUsageRecord<size_t>(size, lower, upper);
    usage_records->push_back(record);
  }
  file.close();
}

void StoreTensorToObject(
    const tflite::gpu::ObjectsAssignment<size_t>& assignment,
    std::string& algorithm) {
  const char* path = std::getenv("TFLITE_CSV_PATH");
  const char* name = std::getenv("TFLITE_TRACE_NAME");

  if (path && name) {
    std::string path_string = std::string(path);
    std::string filename = std::string(name) + "-assignment.txt";
    std::string new_path =
        path_string + "/" + "tflite-csv-out" + "/" + algorithm + "/";
    std::ofstream outfile_out(new_path + filename, std::ios::trunc);

    if (outfile_out.is_open()) {
      outfile_out << assignment.object_ids[0];
      for (size_t i = 1; i < assignment.object_ids.size(); ++i) {
        outfile_out << "," << assignment.object_ids[i];
      }
      outfile_out << std::endl;
      outfile_out << assignment.object_sizes[0];
      for (size_t i = 1; i < assignment.object_sizes.size(); ++i) {
        outfile_out << "," << assignment.object_sizes[i];
      }
      outfile_out.close();
    } else {
      std::cout << "Could not open file: " << new_path << filename << std::endl;
    }
  } else {
    std::cout << "One or both environment variables not found." << std::endl;
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <absolute_path_to_csv_file> <algorithm>" << std::endl;
    return 1;
  }

  std::string filepath = argv[1];
  std::string algorithm = argv[2];

  std::vector<tflite::gpu::TensorUsageRecord<size_t>> usage_records;

  fillTensorUsageRecord(filepath, &usage_records);

  tflite::gpu::ObjectsAssignment<size_t> assignment;

  if (algorithm == "equality") {
    auto start = std::chrono::high_resolution_clock::now();
    if (tflite::gpu::EqualityAssignmentWithHash(usage_records, &assignment) !=
        absl::OkStatus()) {
      std::cerr << "Error with EqualityAssignmentWithHash, exiting..."
                << std::endl;
      return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::ofstream timefile(algorithm + "-times.csv", std::ios::app);
    if (timefile.is_open()) {
      timefile << duration.count() << ",";
      timefile.close();
    } else {
      std::cout << "Could not open file " << algorithm << "-times.csv"
                << std::endl;
    }
  } else if (algorithm == "greedy-breadth") {
    auto start = std::chrono::high_resolution_clock::now();
    if (tflite::gpu::GreedyByBreadthAssignment(usage_records, &assignment) !=
        absl::OkStatus()) {
      std::cerr << "Error with GreedyByBreadthAssignment, exiting..."
                << std::endl;
      return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::ofstream timefile(algorithm + "-times.csv", std::ios::app);
    if (timefile.is_open()) {
      timefile << duration.count() << ",";
      timefile.close();
    } else {
      std::cout << "Could not open file " << algorithm << "-times.csv"
                << std::endl;
    }
  } else if (algorithm == "greedy-size") {
    auto start = std::chrono::high_resolution_clock::now();
    if (tflite::gpu::GreedyBySizeDistPriorityAssignment(
            usage_records, &assignment) != absl::OkStatus()) {
      std::cerr << "Error with GreedyBySizeDistPriorityAssignment, exiting..."
                << std::endl;
      return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::ofstream timefile(algorithm + "-times.csv", std::ios::app);
    if (timefile.is_open()) {
      timefile << duration.count() << ",";
      timefile.close();
    } else {
      std::cout << "Could not open file " << algorithm << "-times.csv"
                << std::endl;
    }
  } else if (algorithm == "greedy-inorder") {
    auto start = std::chrono::high_resolution_clock::now();
    if (tflite::gpu::GreedyInOrderAssignment(usage_records, &assignment) !=
        absl::OkStatus()) {
      std::cerr << "Error with GreedyInOrderAssignment, exiting..."
                << std::endl;
      return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::ofstream timefile(algorithm + "-times.csv", std::ios::app);
    if (timefile.is_open()) {
      timefile << duration.count() << ",";
      timefile.close();
    } else {
      std::cout << "Could not open file " << algorithm << "-times.csv"
                << std::endl;
    }
  } else if (algorithm == "mincostflow") {
    auto start = std::chrono::high_resolution_clock::now();
    if (tflite::gpu::MinCostFlowAssignment(usage_records, &assignment) !=
        absl::OkStatus()) {
      std::cerr << "Error with MinCostFlowAssignment, exiting..." << std::endl;
      return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::ofstream timefile(algorithm + "-times.csv", std::ios::app);
    if (timefile.is_open()) {
      timefile << duration.count() << ",";
      timefile.close();
    } else {
      std::cout << "Could not open file " << algorithm << "-times.csv"
                << std::endl;
    }
  } else if (algorithm == "naive") {
    auto start = std::chrono::high_resolution_clock::now();
    if (tflite::gpu::NaiveAssignment(usage_records, &assignment) !=
        absl::OkStatus()) {
      std::cerr << "Error with NaiveAssignment, exiting..." << std::endl;
      return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::ofstream timefile(algorithm + "-times.csv", std::ios::app);
    if (timefile.is_open()) {
      timefile << duration.count() << ",";
      timefile.close();
    } else {
      std::cout << "Could not open file " << algorithm << "-times.csv"
                << std::endl;
    }
  } else {
    std::cerr << "Invalid algorithm, choose from equality, greedy-breadth, "
                 "greedy-size, greedy-inorder, mincostflow and naive."
              << std::endl;
    return 1;
  }

  tflite::gpu::OffsetsAssignment offsets = ObjectsToOffsets(assignment);

  StoreToCsv(offsets, assignment, usage_records, algorithm);

  //   StoreTensorToObject(assignment, algorithm);
}
