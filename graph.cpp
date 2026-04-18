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
};

// -------------------- CHECK NUMBER --------------------
bool isNumber(string s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!isdigit(c)) return false;
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
            int x = stoi(a[i]);
            int y = stoi(b[i]);

            if (abs(x - y) < 10) score++;
        }
        // categorical comparison
        else {
            if (a[i] == b[i]) score++;
        }
    }

    return score > (n * 0.7); // threshold
}

int main(){
    cout <<"Enter file name: ";
    string filename;
    cin >> filename;
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

    // ---------- PRINT GRAPH ----------
    g.display();

    return 0;
}

