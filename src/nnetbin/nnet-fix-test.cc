#include <ostream>
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

    std::string fix_config;
    po.Register("fix-config", &fix_config,
        "path to the config file of fix strategy");
*/
    std::string fix_config_line;
    po.Register("fix-config-line", &fix_config_line,
		"pass fix-point configuration in text format");

    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_filename = po.GetArg(1),
      input_dimension = po.GetArg(2),
      input_frame_num = po.GetArg(3),
      fix_config = po.GetArg(4);

    std::cout<<"args read"<<std::endl;

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    int dimension;
    std::sscanf(input_dimension.c_str(),"%d",&dimension);    
    int frame_num;
    std::sscanf(input_frame_num.c_str(),"%d",&frame_num);    

    string use_gpu = "yes";
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    NnetFix nnet;
    {
        bool binary_read;
        Input ki(model_filename, &binary_read);
        nnet.Read(ki.Stream(), binary_read);
    }


    if (fix_config_line != "") {
      // Read the fix-point config from cmd line
      nnet.InitFixLine(fix_config_line);
    } 
    else {
      std::cout<<"reading fix_config:"<<fix_config<<std::endl;
      // Read the fix-point config
      nnet.InitFix(fix_config);
    }

    //nnet.ApplyWeightFix();
    //std::cout<<"Nnet fixed"<<std::endl;
    
    CuMatrix<BaseFloat> inmat;
    CuMatrix<BaseFloat> out;

    inmat.Resize(frame_num, dimension);
    inmat.SetRandUniform();

    for (int row = 0; row < inmat.NumRows(); ++row)
    {
      for (int col = 0; col < inmat.NumCols(); ++col)
      {
        inmat(row,col) = inmat(row,col) * 2 - 1;
      }
    }

    nnet.Propagate(inmat, &out);
    const std::vector<CuMatrix<BaseFloat> > pbuf = nnet.PropagateBuffer();
          
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
