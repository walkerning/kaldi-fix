#include <iostream>
#include <string>
#include <tr1/memory>
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
  }
  return 0;
}
