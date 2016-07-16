#include <iostream>
#include <string>
#include <tr1/memory>
#include <assert.h>
#include "fix/fix-strategy.h"
#include "fix/fix-dynamic-fixed-point.h"
#include "fix/fix-null-strategy.h"
using namespace std;

static int _test_status = 0;
#define _FIX_TEST_ASSERT(cond) do {\
    KALDI_ASSERT(cond);            \
    if (!cond) _test_status = 1;   \
  } while (0)

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
    "<DynamicFixedPoint> <DefaultBlobBit> 16 <DefaultParamBit> 16 <BlobIndexBit> 0 32",
  };
  int n = sizeof(cases) / sizeof(string);
       
  for (int i = 0; i < n; i++){
    cout << "Input: " << cases[i] << endl;
    tr1::shared_ptr<kaldi::fix::FixStrategy> strategy = kaldi::fix::FixStrategy::Init(cases[i]);
    kaldi::fix::DynamicFixedPointStrategy* newstrategy = dynamic_cast<kaldi::fix::DynamicFixedPointStrategy*> (strategy.get());
    switch (i) {
    case 2:
      {
        _FIX_TEST_ASSERT(newstrategy->BlobBitNum(0)==16);
        _FIX_TEST_ASSERT(newstrategy->BlobBitNum(1)==8);
        _FIX_TEST_ASSERT(newstrategy->BlobBitNum(2)==8);
        _FIX_TEST_ASSERT(newstrategy->ParamBitNum(0,kaldi::nnet1::Component::kUnknown)==8);
        break;
      }
    case 3:
      {
        _FIX_TEST_ASSERT(newstrategy->ParamBitNum(0,kaldi::nnet1::Component::kUnknown)==16);
        _FIX_TEST_ASSERT(newstrategy->ParamBitNum(1,kaldi::nnet1::Component::kUnknown)==8);
        _FIX_TEST_ASSERT(newstrategy->ParamBitNum(2,kaldi::nnet1::Component::kUnknown)==4);
        _FIX_TEST_ASSERT(newstrategy->BlobBitNum(0)==8);
        break;
      }
    case 4:
      {
        _FIX_TEST_ASSERT(newstrategy->BlobBitNum(2)==4);
        _FIX_TEST_ASSERT(newstrategy->BlobBitNum(4)==16);
        _FIX_TEST_ASSERT(newstrategy->ParamBitNum(0,kaldi::nnet1::Component::kUnknown)==16);
        _FIX_TEST_ASSERT(newstrategy->ParamBitNum(1,kaldi::nnet1::Component::kUnknown)==8);
        _FIX_TEST_ASSERT(newstrategy->ParamBitNum(2,kaldi::nnet1::Component::kUnknown)==4);
        break;
      }
    case 5:
      {
        _FIX_TEST_ASSERT(newstrategy->BlobBitNum(1)==32);
        _FIX_TEST_ASSERT(newstrategy->ParamBitNum(3,kaldi::nnet1::Component::kUnknown)==8);
        _FIX_TEST_ASSERT(newstrategy->ParamBitNum(0,kaldi::nnet1::Component::kUnknown)==4);
        _FIX_TEST_ASSERT(newstrategy->ParamBitNum(2,kaldi::nnet1::Component::kUnknown)==4);
        break;
      }
    case 6:
      {
        _FIX_TEST_ASSERT(newstrategy->BlobBitNum(0)==32);
        _FIX_TEST_ASSERT(newstrategy->BlobBitNum(1)==16);
        _FIX_TEST_ASSERT(newstrategy->ParamBitNum(0,kaldi::nnet1::Component::kUnknown)==16);
        break;
      }
    default:
      {
        printf("no need to test\n");
        break;
      }
    }
  }
  return _test_status;
}
