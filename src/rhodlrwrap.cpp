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
VectorXd getEigenValues(Map<MatrixXd> M) {
    SelfAdjointEigenSolver<MatrixXd> es(M);
    return es.eigenvalues();
}

RcppExport SEXP C_spdinv_eigen ( SEXP X_ )
  {
  using Eigen::Map;
  using Eigen::MatrixXd;
  typedef Eigen::Map<Eigen::MatrixXd> MapMatd;
  const MapMatd X(Rcpp::as<MapMatd>(X_));
  const MatrixXd Xinv(X.inverse());
  return(Rcpp::wrap(Xinv));
  }

SEXP hodlrwrap(SEXP AA,SEXP nn)
  {
  Eigen::MatrixXd denseMatrix;
  Eigen::VectorXd RHS;
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
