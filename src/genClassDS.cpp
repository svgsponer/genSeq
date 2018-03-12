 /**
   genSeqReg.cpp
   Purpose: Generated dataset for sequence regression problems.
   Generates a set of sequences and a corresponding score.
   The score is calculeded based on motifs the sequence contains.

   @author Severin Gsponer (severin.gsponer@insight-centre.org)
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <unordered_map>
#include <functional> //for std::function
#include <algorithm>  //for std::generate_n
#include <future>    //for std::async
#include "cmdline.h"

// typedef std::vector<char> char_array;
// typedef std::vector<std::pair<std::string, double>> motif_array;
// typedef std::vector<std::pair<int, std::string>> dataset;
// typedef std::function<char(void)> char_generator;

template <typename T1, typename T2>
struct pair_1st_cmp: public std::binary_function<bool, T1, T2> {
    bool operator () (const std::pair <T1, T2>& x1, const std::pair<T1, T2> &x2)
    {
        return x1.first > x2.first;
    }
};

using char_array = std::vector<char>;
using motif_array = std::unordered_map<std::string, double>;
using char_generator = std::function<char(void)>;

char_array charset()
{
    //Change this to suit
    return char_array(
                      {   'A','B','C','D','E','F',
                          'G','H','I','J','K',
                          'L','M','N','O','P',
                          'Q','R','S','T','U',
                          'V','W','X','Y','Z',
                          '0','1','2','3','4',
                          '5','6','7','8','9'
                      });
};

std::string random_string(const size_t length, char_generator rand_char )
{
    std::string str(length,0);
    std::generate_n( str.begin(), length, rand_char );
    return str;
}

struct Dataset{
    std::vector<double> scores;
    std::vector<std::string> sequences;
    // std::vector<std::pair<double, std::string>> data;
    void add(double y, std::string seq){
        // std::cout << "Add: " << y << ' ' << seq << "\n";
        scores.push_back(y);
        sequences.push_back(seq);
    }
};

std::ostream& operator<<(std::ostream& os, const Dataset& ds){
    for (std::size_t i = 0, max = ds.scores.size() ; i < max; ++i){
        os << std::showpos << ds.scores[i] << " " << ds.sequences[i] << '\n';
    }
    return os;
};

Dataset create_dataset(const int num_seq,
                 const int seq_length,
                 const motif_array motifs,
                 char_generator rand_char){
    Dataset ds;
    int total_number_rebuild = 0;
    int avg_motifs_counter = 0;
    int min_motifs_counter = motifs.size();
    int max_motifs_counter = 0;
    for (int c=0; c<num_seq; c++) {
        bool has_motif = false;
        std::string seq;
        double y = 0;
        int number_rebuild = -1;
        auto motif_counter = 0;
        do{
            ++number_rebuild;
            seq = random_string(seq_length, rand_char);
            y = 0;
            motif_counter = 0;
            for (auto motif : motifs){
                if(seq.find(motif.first) != std::string::npos){
                    has_motif = true;
                    // std::cout << "Found: " << motif.first << "\n";
                    y += motif.second;
                    ++motif_counter;
                }
            }
        }
        while(!has_motif);
        min_motifs_counter = std::min(min_motifs_counter, motif_counter);
        max_motifs_counter = std::max(max_motifs_counter, motif_counter);
        avg_motifs_counter += motif_counter;
        ds.add(y, seq);
        total_number_rebuild += number_rebuild;
    }
    std::cout << "Average number of motifs per sequence:\t"
              << avg_motifs_counter / num_seq << '\n';
    std::cout << "Min number of motifs per sequence:\t"
              << min_motifs_counter<< '\n';
    std::cout << "Max number of motifs per sequence:\t"
              << max_motifs_counter<< '\n';
    std::cout << "Number of sequence rebuild:\t"
              << total_number_rebuild << '\n';
    return ds;
}

/// Creates random motifs and corresponding weights currently norm(0,5) distributed.
motif_array generate_motifs(const unsigned int motif_length,
                            const unsigned int num_motifs,
                            char_generator rand_char){
    std::mt19937 rngMT (std::random_device{}());
    std::normal_distribution<double> rand_weight(0, 5);
    motif_array motifs;
    while(motifs.size() != num_motifs){
        motifs.emplace(random_string(motif_length, rand_char), rand_weight(rngMT));
    }
    return motifs;
}

double get_treshold(std::vector<double> scores, double ratio){
    std::sort(std::begin(scores), std::end(scores));
    auto t_ind = scores.size() - scores.size() * ratio;
    auto t = scores[t_ind];
    auto it = std::unique(std::begin(scores), std::end(scores));
    std::cout << "Unique scores: " << std::distance(scores.begin(), it) << '\n';
    return t;
}

int main(int argc, char* argv[])
{
    cmdline::parser cmd_parser;

    const auto ch_set = charset();

    std::stringstream oss;
    oss << "Aphabet size (max: " << ch_set.size() << ")";
    std::string alphabet_info_text = oss.str();

    cmd_parser.add<unsigned int>("alphabet_size", 'a', alphabet_info_text, true, 8, cmdline::range(1ul, ch_set.size()));
    cmd_parser.add<unsigned int>("num_seq", 'n', "Number of sequences created", false, 10000);
    cmd_parser.add<unsigned int>("seq_length", 'l', "Sequence length", false, 5000);
    cmd_parser.add<double>("ratio", 'r', "Ratio possitive to negative class", false, 0.5);
    cmd_parser.add<unsigned int>("motiv_length", 'm', "Motiv length", false, 5);
    cmd_parser.add<unsigned int>("num_motifs", 's', "Number of motivs", false, 5);
    cmd_parser.parse_check(argc, argv);
    auto alphabet_size = cmd_parser.get<unsigned int>("alphabet_size");
    auto ratio = cmd_parser.get<double>("ratio");
    auto seq_length = cmd_parser.get<unsigned int>("seq_length");
    auto num_seq = cmd_parser.get<unsigned int>("num_seq");
    auto motif_length = cmd_parser.get<unsigned int>("motiv_length");
    auto num_motifs = cmd_parser.get<unsigned int>("num_motifs");

    if(num_motifs > std::pow(alphabet_size, motif_length)){
        std::cout << "Invalid input: a maximum of " << std::pow(alphabet_size, motif_length)
                  << " unique motifs of length " << motif_length
                  <<  " are possible with with an alphabetsize of "
                  << alphabet_size << std::endl;
        std::exit(-1);
    }

    std::default_random_engine rng(std::random_device{}());

    // String generator
    std::uniform_int_distribution<> dist(0, alphabet_size-1);
    auto randchar = [ ch_set,&dist,&rng ](){return ch_set[ dist(rng) ];};

    // Generate motifs
    motif_array motifs = generate_motifs(motif_length, num_motifs, randchar);

    auto dataset = create_dataset(num_seq, seq_length, motifs, randchar);
    auto threshold = get_treshold(dataset.scores, ratio);

    auto pos = 0;
    auto neg = 0;
    std::for_each(std::begin(dataset.scores), std::end(dataset.scores),
                  [threshold,&pos, &neg](double& score) {
                      if(score <= threshold){
                          ++neg;
                          score = -1;
                      }else{
                          ++pos;
                          score = 1;
                      };});

    std::cout << "Number of sequences:\t" << dataset.scores.size() << '\n'
              << "Positive sequences:\t" << pos << '\n'
              << "Negative sequences:\t" << neg << '\n';



    std::ofstream outfile_train ("toysequences_train");
    outfile_train << dataset;
    // dataset.print_stats();

    // std::ofstream outfile_test ("toysequences_test");
    // outfile_test << test_set;
    // test_set.print_stats();

    // Save correct model
    std::ofstream outfile_correct_model ("toysequences_correctModel");
    outfile_correct_model << "Threshold" << " " <<  threshold << '\n';
    for (auto motif : motifs){
        outfile_correct_model << motif.first << " " <<  motif.second << '\n';
    }


    return 0;
}
