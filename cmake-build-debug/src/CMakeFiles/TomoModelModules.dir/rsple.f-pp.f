# 1 "/Users/ij264/Library/CloudStorage/GoogleDrive-ij264@cam.ac.uk/My Drive/PhD/da380/TomoModel/src/rsple.f"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/Users/ij264/Library/CloudStorage/GoogleDrive-ij264@cam.ac.uk/My Drive/PhD/da380/TomoModel/cmake-build-debug//"
# 1 "/Users/ij264/Library/CloudStorage/GoogleDrive-ij264@cam.ac.uk/My Drive/PhD/da380/TomoModel/src/rsple.f"
      FUNCTION RSPLE(I1,I2,X,Y,Q,S)
C
C C$C$C$C$C$ CALLS ONLY LIBRARY ROUTINES C$C$C$C$C$
C
C   RSPLE RETURNS THE VALUE OF THE FUNCTION Y(X) EVALUATED AT POINT S
C   USING THE CUBIC SPLINE COEFFICIENTS COMPUTED BY RSPLN AND SAVED IN
C   Q.  IF S IS OUTSIDE THE INTERVAL (X(I1),X(I2)) RSPLE EXTRAPOLATES
C   USING THE FIRST OR LAST INTERPOLATION POLYNOMIAL.  THE ARRAYS MUST
C   BE DIMENSIONED AT LEAST - X(I2), Y(I2), AND Q(3,I2).
C
C                                                     -RPB
      DIMENSION X(*),Y(*),Q(3,*)
      DATA I/1/
      II=I2-1
C   GUARANTEE I WITHIN BOUNDS.
      I=MAX0(I,I1)
      I=MIN0(I,II)
C   SEE IF X IS INCREASING OR DECREASING.
      IF(X(I2)-X(I1))1,2,2
C   X IS DECREASING.  CHANGE I AS NECESSARY.
 1    IF(S-X(I))3,3,4
 4    I=I-1
      IF(I-I1)11,6,1
 3    IF(S-X(I+1))5,6,6
 5    I=I+1
      IF(I-II)3,6,7
C   X IS INCREASING.  CHANGE I AS NECESSARY.
 2    IF(S-X(I+1))8,8,9
 9    I=I+1
      IF(I-II)2,6,7
 8    IF(S-X(I))10,6,6
 10   I=I-1
      IF(I-I1)11,6,8
 7    I=II
      GO TO 6
 11   I=I1
C   CALCULATE RSPLE USING SPLINE COEFFICIENTS IN Y AND Q.
 6    H=S-X(I)
      RSPLE=Y(I)+H*(Q(1,I)+H*(Q(2,I)+H*Q(3,I)))
      RETURN
      END
