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
      "Automatically generate a initial fix-strategy file based on random input.\n"
      "Usage: nnet-fix-init [options] <nnet-in> <input-dimension> <input-frame-number>\n"
      "e.g.: nnet-fix-init final.nnet 153 2\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);
    
    std::string fix_config_line;
    po.Register("fix-config-line", &fix_config_line,
		"pass fix-point configuration in text format");

    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_filename = po.GetArg(1),
      input_dimension = po.GetArg(2),
      input_frame_num = po.GetArg(3);

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
      // Init null strategy
      nnet.InitFix("");
    }
    
    CuMatrix<BaseFloat> inmat;
    CuMatrix<BaseFloat> out;

    // generate random input data
    inmat.Resize(frame_num, dimension);
    inmat.SetRandUniform();
    for (int row = 0; row < inmat.NumRows(); ++row)
    {
      for (int col = 0; col < inmat.NumCols(); ++col)
      {
        inmat(row,col) = inmat(row,col) * 2 - 1;
      }
    }

    // confirm fix strategy for model parameters
    nnet.ApplyWeightFix();

    // do the propagation, record the dynamic range of the blob
    nnet.Propagate(inmat, &out);
    const std::vector<CuMatrix<BaseFloat> > pbuf = nnet.PropagateBuffer();
    
    // confirm fix strategy for blob, write fix strategy file
    Output fixconf("/home/xiongzheng/fixconf_init.mod", false);
    nnet.SetupFixStrategy(fixconf.Stream());
    fixconf.Close();

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
