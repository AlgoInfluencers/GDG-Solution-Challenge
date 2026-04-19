#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>

using namespace std;

class Graph{
    int V;
    vector<vector<int>> adjList;

    public:
    Graph(int V){
        this->V = V;
        adjList.resize(V);
    }

    void addEdge(int u, int v){
        adjList[u].push_back(v);
        adjList[v].push_back(u);
    }

    void display(){
        for(int i=0; i<V; i++){
            cout << "Node " << i << ": ";
            for(int j=0; j<adjList[i].size(); j++){
                cout << adjList[i][j] << " ";
            }
            cout << endl;
        }
    }

    void writeEdgesToCSV(const string& filename) {
        ofstream outFile(filename);
        if (!outFile.is_open()) {
            cerr << "Error opening " << filename << " for writing!" << endl;
            return;
        }
        outFile << "source,target\n";
        for (int i = 0; i < V; i++) {
            for (int v : adjList[i]) {
                if (i < v) { // Avoid duplicate edges for undirected graph
                    outFile << i << "," << v << "\n";
                }
            }
        }
        outFile.close();
        cout << "Edges successfully written to " << filename << endl;
    }
};

// -------------------- CHECK NUMBER --------------------
bool isNumber(string s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!isdigit(c) && c != '.') return false; // Allow decimals just in case
    }
    return true;
}

// -------------------- SIMILARITY FUNCTION --------------------
bool isSimilar(vector<string>& a, vector<string>& b) {
    int score = 0;
    int n = a.size();

    for (int i = 0; i < n; i++) {
        // numeric comparison
        if (isNumber(a[i]) && isNumber(b[i])) {
            double x = stod(a[i]);
            double y = stod(b[i]);

            if (abs(x - y) < 10.0) score++;
        }
        // categorical comparison
        else {
            if (a[i] == b[i]) score++;
        }
    }

    return score > (n * 0.7); // threshold
}

int main(int argc, char* argv[]){
    string filename;
    if (argc > 1) {
        filename = argv[1];
    } else {
        cout << "Enter file name: ";
        cin >> filename;
    }
    
   ifstream file(filename);
   if(!file.is_open()){
       cerr << "Error opening file!" << endl;
       return 1;
   }
    vector<string> headers;
    vector<vector<string>> data;
    string line, value;

    // ---------- READ HEADER ----------
    getline(file, line);
    stringstream ss(line);

    while (getline(ss, value, ',')) {
        headers.push_back(value);
    }

    // ---------- READ DATA ----------
    while (getline(file, line)) {
        vector<string> row;
        stringstream ss(line);

        while (getline(ss, value, ',')) {
            row.push_back(value);
        }

        data.push_back(row);
    }

    file.close();

    int n = data.size();
    Graph g(n);

    // ---------- BUILD GRAPH ----------
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (isSimilar(data[i], data[j])) {
                g.addEdge(i, j);
            }
        }
    }

    // ---------- WRITE TO CSV ----------
    g.writeEdgesToCSV("edges.csv");

    return 0;
}

