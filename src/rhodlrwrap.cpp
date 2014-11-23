#include "rhodlrwrap.h"
#include <Rcpp.h>
#include <RcppEigen.h>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::MappedSparseMatrix;
using Eigen::SparseMatrix;

#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

using Eigen::Map;               	// 'maps' rather than copies
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::VectorXd;                  // variable size vector, double precision
using Eigen::SelfAdjointEigenSolver;    // one of the eigenvalue solvers

// [[Rcpp::export]]
RcppExport SEXP getEigenValues(SEXP AA) {
  const Eigen::MatrixXd A( as< Eigen::MatrixXd >(AA) );
  SelfAdjointEigenSolver<MatrixXd> es(A);
  return  Rcpp::wrap(es.eigenvalues());
}

RcppExport SEXP hodlrwrap(SEXP X,SEXP y)
  {
  Eigen::MatrixXd denseMatrix( as<Map<MatrixXd> >(X) );
  Eigen::VectorXd RHS( as<Map<VectorXd> >(y) );
  HODLR_Matrix denseHODLR(denseMatrix, 10 );
  double tol = 10.0;
  denseHODLR.set_LRTolerance(tol);
  Eigen::VectorXd solverSoln;
  solverSoln = denseHODLR.recLU_Solve(RHS);
  //  solverSoln = denseHODLR.extendedSp_Solve(RHS);
  return Rcpp::wrap(solverSoln);
  }

/*
SEXP redSVDwrap(SEXP AA,SEXP nn){
  //  int num=(int)(INTEGER(nn)[0]);
  Rcpp::NumericVector dd(nn);
  int num=(int)dd[0];
   const MappedSparseMatrix<double> A(as<MappedSparseMatrix<double> >(AA));
   REDSVD::RedSVD svA(A, num);
   //   return Rcpp::wrap(svA.matrixV());
   return List::create(Named("V") = Rcpp::wrap(svA.matrixV()),
		       Named("U")=  Rcpp::wrap(svA.matrixU()),
		       Named("D")=  Rcpp::wrap(svA.singularValues()));
}

SEXP redSymwrap(SEXP AA,SEXP nn){
  //  int num=INTEGER(nn)[0];
  Rcpp::NumericVector dd(nn);
  int num=(int)dd[0];
  const MappedSparseMatrix<double> A(as<MappedSparseMatrix<double> >(AA));
  REDSVD::RedSymEigen p(A,num);
  return List::create(Named("eigenValues") = Rcpp::wrap(p.eigenValues()),
		      Named("eigenVectors")= Rcpp::wrap(p.eigenVectors()));
}

SEXP redPCAwrap(SEXP AA,SEXP nn){
  //  int num=INTEGER(nn)[0];
  Rcpp::NumericVector dd(nn);
  int num=(int)dd[0];
   const MappedSparseMatrix<double> A(as<MappedSparseMatrix<double> >(AA));
   REDSVD::RedPCA p(A, num);
   return List::create(Named("principalComponents") = Rcpp::wrap(p.principalComponents()),
		       Named("scores") =  Rcpp::wrap(p.scores()));
}
*/
