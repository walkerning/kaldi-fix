
#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "nnet/nnet-nnet-fix.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
      "Perform forward pass through Neural Network.\n"
      "Usage: nnet-forward [options] <nnet1-in> <feature-rspecifier> <feature-wspecifier>\n"
      "e.g.: nnet-forward final.nnet ark:input.ark ark:output.ark\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);
    /*
    std::string model_filename;
    po.Register("model-filename", &model_filename,
        "name of the unfixed nnet model");

    std::string output_filename;
    po.Register("output-filename", &output_filename,
        "name of output fixed nnet model");
*/
    std::string fix_config;
    po.Register("fix-config", &fix_config,
        "path to the config file of fix strategy");

    std::string fix_config_line;
    po.Register("fix-config-line", &fix_config_line,
		"pass fix-point configuration in text format");

    std::cout<<"begin reading args"<<std::endl;
    po.Read(argc, argv);
    
    std::cout<<"num args: "<<po.NumArgs()<<std::endl<<
        po.GetArg(1)<<std::endl<<
        po.GetArg(2)<<std::endl<<
        po.GetArg(3)<<std::endl;
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_filename = po.GetArg(1),
        output_filename = po.GetArg(2);

    std::cout<<"args read"<<std::endl;

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    std::cout<<"model name: "<<model_filename<<std::endl;
    NnetFix nnet;
    {
        bool binary_read;
        Input ki(model_filename, &binary_read);
        nnet.Read(ki.Stream(), binary_read);
    }

    std::cout<<"model read"<<std::endl;

    if (fix_config_line != "") {
      // Read the fix-point config from cmd line
      nnet.InitFixLine(fix_config_line);
    } else {
      // Read the fix-point config
      nnet.InitFix(fix_config);
    }

    std::cout<<"strategy init"<<std::endl;
    nnet.ApplyWeightFix();
    std::cout<<"Nnet fixed"<<std::endl;
    nnet.Write(output_filename, true);

    std::cout<<"Nnet written"<<std::endl;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
