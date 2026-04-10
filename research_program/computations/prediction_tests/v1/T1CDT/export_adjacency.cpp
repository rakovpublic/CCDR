// export_adjacency.cpp
// Compile: g++ -std=c++20 -O2 export_adjacency.cpp -lCGAL -lgmp -o export_adj
// Usage:   ./export_adj input_triangulation.cgal output_adjacency.txt
//
// NOTE: This is a TEMPLATE. You need to adapt it to match the specific
// CGAL triangulation type used by YOUR version of CDT-plusplus.
// Check the CDT-plusplus source for the typedef of the triangulation.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <fstream>
#include <iostream>
#include <map>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Triangulation;
typedef Triangulation::Cell_handle Cell_handle;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " input.cgal output_adj.txt" << std::endl;
        return 1;
    }

    // Load triangulation
    Triangulation T;
    std::ifstream fin(argv[1]);
    fin >> T;
    fin.close();

    std::cout << "Loaded: " << T.number_of_cells() << " cells, "
              << T.number_of_vertices() << " vertices" << std::endl;

    // Map cells to integer IDs
    std::map<Cell_handle, int> cell_id;
    int id = 0;
    for (auto c = T.finite_cells_begin(); c != T.finite_cells_end(); ++c) {
        cell_id[c] = id++;
    }

    // Export adjacency
    std::ofstream fout(argv[2]);
    fout << "# cell_id neighbour_id_1 neighbour_id_2 ..." << std::endl;
    for (auto c = T.finite_cells_begin(); c != T.finite_cells_end(); ++c) {
        fout << cell_id[c];
        for (int i = 0; i < 4; ++i) {  // 4 neighbours in 3D
            Cell_handle nb = c->neighbor(i);
            if (!T.is_infinite(nb)) {
                fout << " " << cell_id[nb];
            }
        }
        fout << std::endl;
    }
    fout.close();

    std::cout << "Exported " << id << " cells to " << argv[2] << std::endl;
    return 0;
}
