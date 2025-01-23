! -*- f90 -*-
        SUBROUTINE generate_s( RHO, RHOX, RHOY, RHOZ, NBLK, NAT, S )
!*  =====================================================================
!     CALCULATE S FACTOR NUMERICALLY
!     RHO : MATRIX SHAPE (NBLK, NAT)
!     RHOX : MATRIX SHAPE (NBLK, NAT)
!     RHOY : MATRIX SHAPE (NBLK, NAT)
!     RHOZ : MATRIX SHAPE (NBLK, NAT)
!     RETURN: GRAD FACTOR
!     S
!*  =====================================================================
      IMPLICIT NONE
!     Scalar Arguments
      INTEGER NBLK, NAT, I, J
!     Array Arguments
      DOUBLE PRECISION,INTENT(IN) :: RHOX(NBLK,NAT)
      DOUBLE PRECISION,INTENT(IN) :: RHOY(NBLK,NAT)
      DOUBLE PRECISION,INTENT(IN) :: RHOZ(NBLK,NAT)
      DOUBLE PRECISION,INTENT(IN) :: RHO(NBLK,NAT)
      
!     Output Arguments
      DOUBLE PRECISION,INTENT(OUT) :: S(NBLK,NAT)

!*  =====================================================================

      DO I =1,NBLK
        Do J =1,NAT
                S(I,J) = 0d0
        END DO
      END DO

!$OMP PARALLEL DO PRIVATE(I,J) REDUCTION(+: S)
      DO I =1,NBLK
        Do J =1,NAT
                S(I,J) = (RHOX(I,J)*RHOX(I,J)+RHOY(I,J)*RHOY(I,J)+RHOZ(I,J)*RHOZ(I,J))
                S(I,J) = SQRT(S(I,J))*(RHO(I,J)**(-4d0/3d0))
        END DO
      END DO

!$OMP END PARALLEL DO

      RETURN
      END SUBROUTINE

      SUBROUTINE MCSH_R( RHO, STENS, NCB, NA, NSF, NBLK, NAT, NCBT, SHD)
!*  =====================================================================
!     CALCULATE SH FACTOR FOR GIVEN STENS
!     RHO : MATRIX SHAPE (NAT, NBLK)
!     STENS : TENSOR SHAPE (NCBT)
!     RETURN: SH FACTOR
!     SHD
!*  =====================================================================
      IMPLICIT NONE
!     Scalar Arguments
      INTEGER,  INTENT(IN):: NCB, NA, NSF, NBLK, NAT, NCBT
      INTEGER I, J, K, L, IRHO, ISTE
!     Array Arguments
      REAL*8,INTENT(IN) :: RHO(NBLK, NAT)
      REAL*8,INTENT(IN) :: STENS(NCBT)


!     Output Arguments
      DOUBLE PRECISION,INTENT(OUT) :: SHD(NBLK)

!*  =====================================================================
      DO I =1,NBLK
                SHD(I) = 0d0
      END DO


      DO I =1,NBLK
        DO J =1,NCB
          DO K =1,NCB
            DO L = 1,NCB
                IRHO = (NSF+L-1)*NA*NA +(NSF+K-1)*NA +(NSF+J-1) +1
                ISTE = (L-1)*NCB*NCB +(K-1)*NCB +(J-1) +1
!                IF (I == 1.and.J==1.and.K==1.and.L==1) write(*,*) STENS(ISTE), RHO(I,IRHO)
                SHD(I) = SHD(I) + STENS(ISTE)*RHO(I,IRHO)
            END DO
          END DO
        END DO
        SHD(I) = ABS(SHD(I))
!        IF (I == 1) write(*,*) SHD(I)
      END DO


      RETURN
      END SUBROUTINE

        SUBROUTINE merge_shd( SHD1, SHD2, SHD3, NBLK, SHDM )
!*  =====================================================================
!     CALCULATE  MERGED SH FACTOR
!     SHD1 : ARRAY SHAPE (NBLK)
!     SHD2 : ARRAY SHAPE (NBLK)
!     SHD3 : ARRAY SHAPE (NBLK)
!     RETURN: MERGED SH FACTOR
!     SHDM
!*  =====================================================================
      IMPLICIT NONE
!     Scalar Arguments
      INTEGER NBLK, I
!     Array Arguments
      DOUBLE PRECISION,INTENT(IN) :: SHD1(NBLK)
      DOUBLE PRECISION,INTENT(IN) :: SHD2(NBLK)
      DOUBLE PRECISION,INTENT(IN) :: SHD3(NBLK)
      
!     Output Arguments
      DOUBLE PRECISION,INTENT(OUT) :: SHDM(NBLK)

!*  =====================================================================


!$OMP PARALLEL DO PRIVATE(I) REDUCTION(+: SHDM)
      DO I =1,NBLK
            SHDM(I) = 0d0
            SHDM(I) = SQRT(SHD1(I)*SHD1(I)+SHD2(I)*SHD2(I)+SHD3(I)*SHD3(I))
      END DO

!$OMP END PARALLEL DO

      RETURN
      END SUBROUTINE



