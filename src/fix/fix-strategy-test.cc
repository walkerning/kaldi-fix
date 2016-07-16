#include <iostream>
#include <string>
#include <tr1/memory>
#include <assert.h>
#include "fix/fix-strategy.h"
#include "fix/fix-dynamic-fixed-point.h"
#include "fix/fix-null-strategy.h"
using namespace std;

int main()
{
  string cases[] = {
    "",
    "    ",
    // Wrong input will raise error
    // "Dynamic"
    // "<Dynamic>",
    // "<DynamicFixedPoint> <BlobInBit> 0 16",
    "<DynamicFixedPoint> <BlobIndexBit> 0 16 <BlobIndexBit> 1 8 <BlobIndexBit> 2 8",
    "<DynamicFixedPoint> <ParamIndexBit> 0 16 <ParamIndexBit> 1 8 <ParamIndexBit> 2 4",
    "<DynamicFixedPoint> <BlobIndexBit> 2 4 <ParamIndexBit> 0 16 <ParamIndexBit> 1 8 <BlobIndexBit> 4 16 <ParamIndexBit> 2 4",
    "<DynamicFixedPoint> <BlobIndexBit> 1 32 <ParamIndexBit> 3 8 <ParamIndexBit> 0 4 <ParamIndexBit> 2 4",
  };
  int n = sizeof(cases) / sizeof(string);
	
  for (int i = 0; i < n; i++){
    cout << "Input: " << cases[i] << endl;
    tr1::shared_ptr<kaldi::fix::FixStrategy> strategy = kaldi::fix::FixStrategy::Init(cases[i]);
    kaldi::fix::DynamicFixedPointStrategy* newstrategy = dynamic_cast<kaldi::fix::DynamicFixedPointStrategy*> (strategy.get());
    switch (i) {
    case 2:
      {
      assert(newstrategy->BlobBitNum(0)==16);
      assert(newstrategy->BlobBitNum(1)==8);
      assert(newstrategy->BlobBitNum(2)==8);
      assert(newstrategy->ParamBitNum(0,kaldi::nnet1::Component::kUnknown)==8);
      break;
      }
    case 3:
      {
	assert(newstrategy->ParamBitNum(0,kaldi::nnet1::Component::kUnknown)==16);
        assert(newstrategy->ParamBitNum(1,kaldi::nnet1::Component::kUnknown)==8);
        assert(newstrategy->ParamBitNum(2,kaldi::nnet1::Component::kUnknown)==4);
        assert(newstrategy->BlobBitNum(0)==8);
	break;
      }
    case 4:
      {
        assert(newstrategy->BlobBitNum(2)==4);
        assert(newstrategy->BlobBitNum(4)==16);
        assert(newstrategy->ParamBitNum(0,kaldi::nnet1::Component::kUnknown)==16);
        assert(newstrategy->ParamBitNum(1,kaldi::nnet1::Component::kUnknown)==8);
        assert(newstrategy->ParamBitNum(2,kaldi::nnet1::Component::kUnknown)==4);
	break;
      }
    case 5:
      {
        assert(newstrategy->BlobBitNum(1)==32);
        assert(newstrategy->ParamBitNum(3,kaldi::nnet1::Component::kUnknown)==8);
        assert(newstrategy->ParamBitNum(0,kaldi::nnet1::Component::kUnknown)==4);
        assert(newstrategy->ParamBitNum(2,kaldi::nnet1::Component::kUnknown)==4);
	break;
      }
    default:
      {
	printf("no need to test\n");
	break;
      }
    }

  }
  return 0;
}
