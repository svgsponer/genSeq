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

using char_array = std::vector<char>;
using motif_map = std::unordered_map<std::string, double>;
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
}

std::string random_string(const size_t length, char_generator rand_char )
{
    std::string str(length,0);
    std::generate_n( str.begin(), length, rand_char );
    return str;
}

struct Dataset{
    std::vector<double> scores;
    std::vector<std::string> sequences;

    void add(double y, std::string seq){
        scores.push_back(y);
        sequences.push_back(seq);
    }

    std::size_t size(){
        return scores.size();
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
                 const motif_map motifs,
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
        while(!has_motif){
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
        min_motifs_counter = std::min(min_motifs_counter, motif_counter);
        max_motifs_counter = std::max(max_motifs_counter, motif_counter);
        avg_motifs_counter += motif_counter;
        ds.add(y, seq);
        total_number_rebuild += number_rebuild;
        printf("%d/%d\r", c, num_seq);
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
motif_map generate_motifs(const unsigned int motif_length,
                            const unsigned int num_motifs,
                            char_generator rand_char){
    std::mt19937 rngMT (std::random_device{}());
    std::normal_distribution<double> rand_weight(0, 5);
    motif_map motifs;
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

std::tuple<Dataset, Dataset> split_dataset(Dataset& ds, double ratio){

    int dist = ds.scores.size() * ratio;
    Dataset training_set;
    std::copy(ds.scores.begin() + dist, std::end(ds.scores), std::back_inserter(training_set.scores));
    std::copy(ds.sequences.begin() + dist, std::end(ds.sequences), std::back_inserter(training_set.sequences));

    Dataset test_set;
    std::copy(ds.scores.begin(), ds.scores.begin() + dist, std::back_inserter(test_set.scores));
    std::copy(ds.sequences.begin(), ds.sequences.begin() + dist, std::back_inserter(test_set.sequences));

    return std::make_tuple(training_set, test_set);
}

void print_stats(Dataset& ds){
    auto pos = std::count(std::begin(ds.scores), std::end(ds.scores), 1);
    auto neg = ds.size() - pos;
    std::cout << "Number of sequences:\t" << ds.scores.size() << '\n'
              << "Positive sequences:\t" << pos << '\n'
              << "Negative sequences:\t" << neg << '\n';
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
    cmd_parser.add<double>("ttratio", 't', "Ratio between test and trainingset size", false, 0.2);
    cmd_parser.add<unsigned int>("motiv_length", 'm', "Motiv length", false, 5);
    cmd_parser.add<unsigned int>("num_motifs", 's', "Number of motivs", false, 5);
    cmd_parser.parse_check(argc, argv);
    auto alphabet_size = cmd_parser.get<unsigned int>("alphabet_size");
    auto ratio = cmd_parser.get<double>("ratio");
    auto tt_ratio = cmd_parser.get<double>("ttratio");
    auto seq_length = cmd_parser.get<unsigned int>("seq_length");
    auto num_seq = cmd_parser.get<unsigned int>("num_seq");
    auto motif_length = cmd_parser.get<unsigned int>("motiv_length");
    auto num_motifs = cmd_parser.get<unsigned int>("num_motifs");

    // Check for input errors
    if(num_motifs > std::pow(alphabet_size, motif_length)){
        std::cout << "Invalid input: a maximum of " << std::pow(alphabet_size, motif_length)
                  << " unique motifs of length " << motif_length
                  <<  " are possible with with an alphabetsize of "
                  << alphabet_size << std::endl;
        std::exit(-1);
    }

    if(seq_length < motif_length){
        std::cout << "Invalid input: sequence length must be at least as long as the generated motifs.\n"
                  << "motif length: " << motif_length << '\n'
                  << "sequence length: " << seq_length
                  << std::endl;
        std::exit(-1);
    }

    if(num_motifs <= 1){
        std::cout << "Invalid input: at least two motifs required to produce different scores/classes.\n"
                  << "num_motifs: " << num_motifs
                  << std::endl;
        std::exit(-1);
    }

    std::default_random_engine rng(std::random_device{}());

    // String generator
    std::uniform_int_distribution<> dist(0, alphabet_size-1);
    auto randchar = [ ch_set,&dist,&rng ](){return ch_set[ dist(rng) ];};

    // Generate motifs
    motif_map motifs = generate_motifs(motif_length, num_motifs, randchar);

    auto dataset = create_dataset(num_seq, seq_length, motifs, randchar);
    auto threshold = get_treshold(dataset.scores, ratio);

    std::for_each(std::begin(dataset.scores), std::end(dataset.scores),
                  [threshold](double& score) {
                      score = (score <= threshold) ? -1 : +1;});


    // Calculate and print stats
    std::cout << "Complete set:"  << "\n";
    print_stats(dataset);

    Dataset train_set;
    Dataset test_set;
    std::tie(train_set, test_set) = split_dataset(dataset, tt_ratio);

    std::cout << "Training set:"  << "\n";
    print_stats(train_set);
    std::cout << "Test set:"  << "\n";
    print_stats(test_set);

    // Save dataset
    std::ofstream outfile_train ("toysequences_train");
    outfile_train << train_set;

    std::ofstream outfile_test ("toysequences_test");
    outfile_test << test_set;

    // Save correct model
    std::ofstream outfile_correct_model ("toysequences_correctModel");
    outfile_correct_model << "Threshold" << " " <<  threshold << '\n';
    for (auto motif : motifs){
        outfile_correct_model << motif.first << " " <<  motif.second << '\n';
    }


    return 0;
}
